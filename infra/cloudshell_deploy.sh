#!/usr/bin/env bash
# Deploy the Wafer Yield API to Azure Container Apps from Azure Cloud Shell (Bash).
#
# Why Cloud Shell: my university tenant blocks Azure CLI and Terraform sign-in from a laptop,
# but Cloud Shell is already signed in and has both.
#
#   git clone https://github.com/navyathag13-ui/wafer-yield.git && cd wafer-yield/infra
#   ALERT_EMAIL=you@example.com ./cloudshell_deploy.sh
#
# Steps: create the registry, build the image inside Azure (az acr build, no local Docker),
# then create the Container Apps environment and app. Shows each plan and asks before applying.
# Tear down with ./cloudshell_destroy.sh (the resource group itself is left alone).
set -euo pipefail

: "${ALERT_EMAIL:?Set ALERT_EMAIL to the address that should receive budget alerts}"
RG="${RG:-rg-pricecheck}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"

export ARM_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"
export TF_VAR_alert_email="$ALERT_EMAIL"
export TF_VAR_resource_group_name="$RG"
az group show -n "$RG" --query name -o tsv >/dev/null
az provider register --namespace Microsoft.App --wait >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait >/dev/null

terraform init -input=false

echo "== Step 1: registry, logs and environment (no app yet)"
terraform plan -input=false -var image_tag= -out=tfplan1
read -r -p "Apply step 1? (type 'yes') " a; [ "$a" = "yes" ] || exit 0
terraform apply -input=false tfplan1

ACR="$(terraform output -raw registry_name)"
echo "== Step 2: build image wafer-yield:$TAG inside $ACR"
az acr build -r "$ACR" -t "wafer-yield:$TAG" ..

echo "== Step 3: the app"
terraform plan -input=false -var image_tag="$TAG" -out=tfplan2
echo "Cost: about \$0.06/hour with one warm replica (1 vCPU, 2 GiB) plus about \$0.17/day for the registry."
read -r -p "Apply step 3? (type 'yes') " a; [ "$a" = "yes" ] || exit 0
terraform apply -input=false tfplan2

URL="$(terraform output -raw api_url)"
echo "API: $URL"
curl -fsS "$URL/health" && echo
