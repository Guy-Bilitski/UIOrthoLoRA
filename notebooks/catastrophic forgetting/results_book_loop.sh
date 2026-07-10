#!/usr/bin/env bash
# Auto-update loop for the results book.
# Every 30 min: regenerate results_book/; if anything changed, commit + push.
# Launch detached:
#   setsid bash results_book_loop.sh >> logs/results_book.log 2>&1 < /dev/null &
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

INTERVAL=1800

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] regenerating results book"
    python3 "$DIR/results_book.py" || echo "WARN: generator failed rc=$?"

    # guard: results_book must not be gitignored (someone may re-tighten .gitignore)
    if git check-ignore -q results_book/README.md 2>/dev/null; then
        gitroot="$(git rev-parse --show-toplevel)"
        {
            echo ""
            echo "# results book (auto-generated presentation tables) must be tracked"
            echo '!notebooks/catastrophic forgetting/results_book/'
            echo '!notebooks/catastrophic forgetting/results_book/*.md'
        } >> "$gitroot/.gitignore"
        echo "NOTE: results_book/ was gitignored; appended negation to $gitroot/.gitignore"
    fi

    if [ -n "$(git status --short results_book/)" ]; then
        ts="$(date '+%Y-%m-%d %H:%M')"
        git add results_book/
        git commit -m "results book auto-update $ts" -- results_book/ \
            && git push \
            && echo "[$ts] committed + pushed" \
            || echo "WARN: commit/push failed"
    else
        echo "no changes"
    fi
    sleep "$INTERVAL"
done
