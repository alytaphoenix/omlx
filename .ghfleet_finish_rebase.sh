#!/bin/sh
set -eu
f=tests/test_cluster_routes.py
perl -0pi -e 's/def resolve_cluster_model_id\(self, model\):/def resolve_cluster_model_id(self, model, *, text_only=False):/g' "$f"
git add "$f"
git commit --amend --no-edit
