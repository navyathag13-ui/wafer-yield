output "registry" {
  value = azurerm_container_registry.acr.login_server
}

output "registry_name" {
  value = azurerm_container_registry.acr.name
}

output "api_url" {
  value = var.image_tag == "" ? null : "https://${azurerm_container_app.api[0].ingress[0].fqdn}"
}
