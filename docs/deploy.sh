#!/bin/bash

# Exit on error
set -e

# Save current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Ensure we're on main branch
git checkout main

# Build the site
echo "Building site..."
JEKYLL_ENV=production bundle exec jekyll build

# Create a temporary directory
TEMP_DIR=$(mktemp -d)

# Copy _site contents to temp directory
echo "Copying _site to temporary directory..."
cp -r _site/* "$TEMP_DIR/"

# Switch to gh-pages branch
echo "Switching to gh-pages branch..."
if ! git show-ref --verify --quiet refs/heads/gh-pages; then
    echo "Creating gh-pages branch..."
    git checkout --orphan gh-pages
    git reset --hard
    rm -rf *
    git commit --allow-empty -m "Initial gh-pages commit"
else
    git checkout gh-pages
    git reset --hard
    rm -rf *
fi

# Copy from temp directory
echo "Copying from temporary directory..."
cp -r "$TEMP_DIR"/* .

# Cleanup temp directory
rm -rf "$TEMP_DIR"

# Add and commit changes
echo "Committing changes..."
git add -A
git commit -m "Deploy site $(date)"

# Push to GitHub
echo "Pushing to GitHub..."
git push origin gh-pages --force

# Switch back to original branch
git checkout "$CURRENT_BRANCH"

echo "Deployment complete!"
