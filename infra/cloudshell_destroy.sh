#!/usr/bin/env bash
# Delete everything cloudshell_deploy.sh created (registry, logs, environment, app, budget).
set -euo pipefail
: "${ALERT_EMAIL:=unused@example.com}"
export ARM_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
export TF_VAR_alert_email="$ALERT_EMAIL"
export TF_VAR_resource_group_name="${RG:-rg-pricecheck}"
terraform destroy -input=false -var image_tag="${TAG:-x}"
