#!/bin/bash
set -eu

[ "$GH_PASSWORD" ] || exit 12

head=$(git rev-parse HEAD)

git clone -b gh-pages "https://dfki-ric:$GH_PASSWORD@github.com/$GITHUB_REPOSITORY.git" gh-pages
cp -R doc/build/html/* gh-pages/
cd gh-pages
git rm -r *
git add *
touch .nojekyll
git add .nojekyll
if git diff --staged --quiet; then
  echo "$0: No changes to commit."
  exit 0
fi

if ! git config user.name; then
    git config user.name 'github-actions'
    git config user.email 'afabisch@googlemail.com'
fi

git commit -a -m "CI: Update docs for $head"
git push
