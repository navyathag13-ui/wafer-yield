variable "resource_group_name" {
  description = "Existing resource group to deploy into. Terraform reads it, it does not create or delete it."
  type        = string
  default     = "rg-pricecheck"
}

variable "location" {
  description = "Region. This subscription only allows some regions (northcentralus, westus, francecentral, norwayeast, denmarkeast)."
  type        = string
  default     = "northcentralus"
}

variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "waferyield"
}

variable "image_tag" {
  description = "Tag of the wafer-yield image in the registry (built by `az acr build`). Empty means create only the registry."
  type        = string
  default     = ""
}

variable "min_replicas" {
  description = "Replicas kept warm. 1 avoids cold starts; 0 scales to zero when idle (cheapest)."
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Upper bound for HTTP autoscaling."
  type        = number
  default     = 5
}

variable "concurrent_requests_per_replica" {
  description = "Scale out when a replica has more than this many requests in flight. The local load test showed one 1-vCPU replica saturating around 10."
  type        = number
  default     = 10
}

variable "alert_email" {
  description = "Address for the budget alert. Passed at deploy time, never committed."
  type        = string
}

variable "budget_amount_usd" {
  description = "Monthly budget alert for the wafer resources, in USD."
  type        = number
  default     = 10
}
