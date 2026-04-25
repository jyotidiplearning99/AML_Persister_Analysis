#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix the LOGO-patched Transformer script in place.

The patched file currently has THREE different bugs:

  Bug 1 (line 288):  skf.split(X, y, groups=sids)   inside a function where
                     sids is not in scope. This is the NameError you hit.

  Bug 2 (line 475):  sss.split(X, y, groups=sids)   on a StratifiedShuffleSplit
                     that should NOT have groups= (held-out test selection).
                     Silently ignored by sklearn but wrong intent.

  Bug 3 (line 523):  val_split.split(X_train_p, y_train, groups=sids)
                     where sids has length 32,342 but X_train_p has length 25,873.
                     Will raise a length-mismatch error if execution gets here.

The fixes:

  Fix 1: Add `sids` parameter to stratified_kfold_validation function signature.
         Line 276: def stratified_kfold_validation(X, y, n_folds=3):
              ->   def stratified_kfold_validation(X, y, sids, n_folds=3):

  Fix 2: Pass sids[train_idx] at the call site.
         Line 487: stratified_kfold_validation(X_train, y_train, n_folds=3)
              ->   stratified_kfold_validation(X_train, y_train, sids[train_idx], n_folds=3)

  Fix 3: Remove the bogus groups=sids from the held-out test split.
         Line 475: next(sss.split(X, y, groups=sids))
              ->   next(sss.split(X, y))

  Fix 4: Slice sids to match X_train_p length at the val_split site.
         Line 523: val_split.split(X_train_p, y_train, groups=sids)
              ->   val_split.split(X_train_p, y_train, groups=sids[train_idx])

This script edits in place. Run on the BROKEN
production_transformer_LOGO_25042026.py to fix it.

Usage:
    python fix_logo_sids.py production_transformer_LOGO_25042026.py --check-only
    python fix_logo_sids.py production_transformer_LOGO_25042026.py
"""

import argparse
import sys
from pathlib import Path


# Anchor lines and expected substrings — verified against actual file via grep.
EXPECTED = {
    276: 'def stratified_kfold_validation(X, y, n_folds=3):',
    288: 'for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y, groups=sids), 1):',
    475: 'train_idx, test_idx = next(sss.split(X, y, groups=sids))',
    487: '    threshold, oof_probs = stratified_kfold_validation(X_train, y_train, n_folds=3)',
    523: '    tr_idx, val_idx = next(val_split.split(X_train_p, y_train, groups=sids))',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='Path to broken LOGO file (will be edited in place)')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--no-backup', action='store_true',
                    help="Don't write a .bak file (default: write backup)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(2)

    src_lines = path.read_text().splitlines(keepends=True)
    n_lines = len(src_lines)
    print(f"File: {path} ({n_lines} lines)")
    print()

    # Verify all anchors
    ok_all = True
    print("Anchor verification:")
    for ln, exp in EXPECTED.items():
        if ln - 1 >= n_lines:
            print(f"  ✗ line {ln}: file is too short")
            ok_all = False
            continue
        actual = src_lines[ln - 1].rstrip('\n').rstrip('\r')
        # Compare with rstripped lines so trailing whitespace doesn't matter
        actual_stripped = actual.rstrip()
        exp_stripped = exp.rstrip()
        # Be lenient about leading whitespace in some anchors
        if exp_stripped in actual or actual_stripped == exp_stripped:
            print(f"  ✓ line {ln}: {actual_stripped!r}")
        else:
            ok_all = False
            print(f"  ✗ line {ln}: mismatch")
            print(f"      expected: {exp_stripped!r}")
            print(f"      actual:   {actual_stripped!r}")
    print()

    if not ok_all:
        print("ABORT: anchors did not match the expected file layout.", file=sys.stderr)
        print("       The file may have been edited since the patch was applied.",
              file=sys.stderr)
        print("       Show me the relevant lines (sed -n) and I'll update.",
              file=sys.stderr)
        sys.exit(3)

    # Apply fixes
    new_lines = src_lines.copy()

    # Fix 1: line 276 - add sids parameter to function signature
    new_lines[275] = new_lines[275].replace(
        'def stratified_kfold_validation(X, y, n_folds=3):',
        'def stratified_kfold_validation(X, y, sids, n_folds=3):'
    )

    # Fix 2: line 487 - pass sids[train_idx] at call site
    new_lines[486] = new_lines[486].replace(
        'stratified_kfold_validation(X_train, y_train, n_folds=3)',
        'stratified_kfold_validation(X_train, y_train, sids[train_idx], n_folds=3)'
    )

    # Fix 3: line 475 - remove groups=sids from held-out test split
    new_lines[474] = new_lines[474].replace(
        'next(sss.split(X, y, groups=sids))',
        'next(sss.split(X, y))'
    )

    # Fix 4: line 523 - slice sids to training subset for val split
    new_lines[522] = new_lines[522].replace(
        'val_split.split(X_train_p, y_train, groups=sids)',
        'val_split.split(X_train_p, y_train, groups=sids[train_idx])'
    )

    # Print post-patch lines for verification
    print("Post-fix lines:")
    for ln in (276, 288, 475, 487, 523):
        print(f"  Line {ln}: {new_lines[ln-1].rstrip()!r}")
    print()

    # Sanity: line 288 (inside the function) should still have groups=sids — that's
    # what we WANT, because now sids is a parameter of that function.
    if 'groups=sids' not in new_lines[287]:
        print("ERROR: line 288 should still have groups=sids (it's the parameter now). Aborting.",
              file=sys.stderr)
        sys.exit(4)

    if args.check_only:
        print("--check-only: not writing.")
        sys.exit(0)

    # Backup
    if not args.no_backup:
        backup_path = path.with_suffix(path.suffix + '.bak')
        backup_path.write_text(''.join(src_lines))
        print(f"Backup: {backup_path}")

    # Write
    path.write_text(''.join(new_lines))
    print(f"Wrote: {path}")
    print()
    print("Verification:")
    print(f"  grep -n 'sids' {path}")
    print()
    print("Expected sids occurrences after fix:")
    print("  - line 7:   header comment")
    print("  - line 276: function parameter (def ... sids, ...)")
    print("  - line 288: skf.split(X, y, groups=sids)         [in-function ref]")
    print("  - line 468: sids = np.array(sid_list)             [original construction]")
    print("  - line 475: next(sss.split(X, y))                 [should NOT have sids any more]")
    print("  - line 487: stratified_kfold_validation(..., sids[train_idx], ...)  [fixed call]")
    print("  - line 523: val_split.split(..., groups=sids[train_idx])           [sliced]")


if __name__ == '__main__':
    main()
