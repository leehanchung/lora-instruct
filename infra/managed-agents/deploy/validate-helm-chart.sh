#!/bin/bash
# Helm chart validation script

set -e

CHART_PATH="./helm/managed-agents"
ERRORS=0

echo "=========================================="
echo "Validating Helm Chart"
echo "=========================================="
echo ""

# Check if Helm is installed
if ! command -v helm &> /dev/null; then
    echo "ERROR: Helm is not installed"
    exit 1
fi

echo "[1] Checking chart structure..."
if [ -f "$CHART_PATH/Chart.yaml" ]; then
    echo "  ✓ Chart.yaml found"
else
    echo "  ✗ Chart.yaml missing"
    ((ERRORS++))
fi

if [ -f "$CHART_PATH/values.yaml" ]; then
    echo "  ✓ values.yaml found"
else
    echo "  ✗ values.yaml missing"
    ((ERRORS++))
fi

if [ -d "$CHART_PATH/templates" ]; then
    TEMPLATE_COUNT=$(find "$CHART_PATH/templates" -name "*.yaml" -o -name "*.tpl" | wc -l)
    echo "  ✓ templates/ directory found ($TEMPLATE_COUNT files)"
else
    echo "  ✗ templates/ directory missing"
    ((ERRORS++))
fi

echo ""
echo "[2] Linting chart..."
if helm lint "$CHART_PATH" > /dev/null 2>&1; then
    echo "  ✓ Chart linting passed"
else
    echo "  ✗ Chart linting failed"
    helm lint "$CHART_PATH"
    ((ERRORS++))
fi

echo ""
echo "[3] Validating templates..."
if helm template managed-agents "$CHART_PATH" > /dev/null 2>&1; then
    echo "  ✓ Template rendering successful"
else
    echo "  ✗ Template rendering failed"
    helm template managed-agents "$CHART_PATH"
    ((ERRORS++))
fi

echo ""
echo "[4] Checking required files..."
REQUIRED_FILES=(
    "$CHART_PATH/Chart.yaml"
    "$CHART_PATH/values.yaml"
    "$CHART_PATH/templates/_helpers.tpl"
    "$CHART_PATH/templates/api-deployment.yaml"
    "$CHART_PATH/templates/postgres-statefulset.yaml"
    "$CHART_PATH/templates/redis-deployment.yaml"
    "$CHART_PATH/templates/rbac.yaml"
    "$CHART_PATH/templates/networkpolicy.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $(basename $file)"
    else
        echo "  ✗ $(basename $file) MISSING"
        ((ERRORS++))
    fi
done

echo ""
echo "[5] Checking configuration files..."
CONFIG_FILES=(
    "./pyproject.toml"
    "./docker-compose.yaml"
    "./Dockerfile"
    "./Dockerfile.runtime"
    "./.env.example"
    "./Makefile"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $(basename $file)"
    else
        echo "  ✗ $(basename $file) MISSING"
        ((ERRORS++))
    fi
done

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "Validation PASSED ✓"
    echo "=========================================="
    echo ""
    echo "Quick Start:"
    echo "  1. Development: docker-compose up -d"
    echo "  2. Kubernetes:  helm install managed-agents ./helm/managed-agents"
    exit 0
else
    echo "Validation FAILED ✗ ($ERRORS errors)"
    echo "=========================================="
    exit 1
fi
