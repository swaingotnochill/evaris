#!/bin/bash
# Usage: ./publish.sh [test|prod]

set -e 

cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TARGET=${1:-test}

echo -e "${GREEN}=== evaris-py Publishing Script ===${NC}"
echo ""

if [[ "$TARGET" != "test" && "$TARGET" != "prod" ]]; then
    echo -e "${RED}Error: Target must be 'test' or 'prod'${NC}"
    echo "Usage: ./publish.sh [test|prod]"
    exit 1
fi

if [[ "$TARGET" == "prod" ]]; then
    echo -e "${YELLOW}WARNING: Publishing to PRODUCTION PyPI!${NC}"
    read -p "Are you sure? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

VERSION=$(grep "^version" pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo -e "${GREEN}Version: ${VERSION}${NC}"
echo ""

echo -e "${GREEN}Step 1: Running pre-flight checks...${NC}"

echo "  - Running tests..."
if ! pytest --cov=evaris -q; then
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi

echo "  - Checking code formatting..."
if ! black evaris tests --check -q; then
    echo -e "${YELLOW}Code not formatted. Running black...${NC}"
    black evaris tests
fi

echo "  - Linting code..."
if ! ruff check evaris tests; then
    echo -e "${RED}Linting failed!${NC}"
    exit 1
fi

echo "  - Type checking..."
if ! mypy evaris; then
    echo -e "${RED}Type checking failed!${NC}"
    exit 1
fi

echo -e "${GREEN}All checks passed!${NC}"
echo ""

echo -e "${GREEN}Step 2: Cleaning previous builds...${NC}"
rm -rf dist/ build/ *.egg-info/
echo "  Cleaned: dist/, build/, *.egg-info/"
echo ""

echo -e "${GREEN}Step 3: Building package...${NC}"
python -m build
echo "  Built: dist/evaris-${VERSION}-py3-none-any.whl"
echo "  Built: dist/evaris-${VERSION}.tar.gz"
echo ""

echo -e "${GREEN}Step 4: Checking package with twine...${NC}"
if ! twine check dist/*; then
    echo -e "${RED}Package check failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Package check passed!${NC}"
echo ""

if [[ "$TARGET" == "test" ]]; then
    echo -e "${GREEN}Step 5: Uploading to TestPyPI...${NC}"
    twine upload --repository testpypi dist/*
    echo ""
    echo -e "${GREEN}Upload successful!${NC}"
    echo ""
    echo "Test installation:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ evaris"
    echo ""
    echo "View at: https://test.pypi.org/project/evaris/${VERSION}/"
else
    echo -e "${GREEN}Step 5: Uploading to PyPI...${NC}"
    twine upload dist/*
    echo ""
    echo -e "${GREEN}Upload successful!${NC}"
    echo ""
    echo "Installation:"
    echo "  pip install evaris"
    echo ""
    echo "View at: https://pypi.org/project/evaris/${VERSION}/"
    echo ""
    echo -e "${YELLOW}Don't forget to:${NC}"
    echo "  1. Create git tag: git tag -a v${VERSION} -m 'Release v${VERSION}'"
    echo "  2. Push tag: git push origin v${VERSION}"
    echo "  3. Create GitHub release: https://github.com/swaingotnochill/evaris/releases/new"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
