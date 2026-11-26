# Wafer Yield API on Azure Container Apps: a registry for the image, a Log Analytics workspace for
# container logs, a Container Apps environment, and the app itself with HTTP autoscaling and health probes.

data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.name}-logs"
  location            = var.location
  resource_group_name = data.azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_registry" "acr" {
  name                = "${var.name}${random_string.suffix.result}"
  location            = var.location
  resource_group_name = data.azurerm_resource_group.main.name
  sku                 = "Basic"
  admin_enabled       = true # Container Apps pulls with these credentials, stored as an app secret
}

resource "azurerm_container_app_environment" "env" {
  name                       = "${var.name}-env"
  location                   = var.location
  resource_group_name        = data.azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

resource "azurerm_container_app" "api" {
  count                        = var.image_tag == "" ? 0 : 1
  name                         = "${var.name}-api"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = data.azurerm_resource_group.main.name
  revision_mode                = "Single"

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.acr.admin_password
  }

  registry {
    server               = azurerm_container_registry.acr.login_server
    username             = azurerm_container_registry.acr.admin_username
    password_secret_name = "acr-password"
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    http_scale_rule {
      name                = "http-concurrency"
      concurrent_requests = tostring(var.concurrent_requests_per_replica)
    }

    container {
      name   = "api"
      image  = "${azurerm_container_registry.acr.login_server}/wafer-yield:${var.image_tag}"
      cpu    = 1.0
      memory = "2Gi"

      startup_probe {
        transport               = "HTTP"
        port                    = 8000
        path                    = "/health"
        interval_seconds        = 5
        failure_count_threshold = 20 # model files load on the first /health call
      }
      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
    }
  }
}

resource "azurerm_consumption_budget_resource_group" "budget" {
  name              = "${var.name}-monthly-budget"
  resource_group_id = data.azurerm_resource_group.main.id
  amount            = var.budget_amount_usd
  time_grain        = "Monthly"

  time_period {
    start_date = "2026-11-01T00:00:00Z"
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    contact_emails = [var.alert_email]
  }
}
