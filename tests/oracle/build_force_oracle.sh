#!/usr/bin/env bash
# Build instrumented copies of the legacy Fortran that dump per-particle forces
# for the initial configuration and then stop. These are the oracles for the
# tier-2 differential test.
#
# They are COPIES -- the pristine tree in legacy/fortran_bd/ is never modified
# or built in place.
#
# Two variants are produced, because the two force contributions have very
# different conditioning:
#
#   full/  all forces (pair + dielectrophoretic trap)
#   pair/  DEP accumulation removed, leaving DLVO + dipole only
#
# The pair forces are well conditioned and must agree with the port to
# round-off. The DEP term cannot: it is a forward difference with h = 1e-3 nm on
# a quantity of order 1, whose terms differ by only ~1e-10 relatively. That
# cancellation burns ~10 of FP64's ~16 digits, leaving ~2e-6 of relative
# accuracy that is compiler-dependent and not reproducible across languages.
# Isolating the pair forces is what makes a 1e-12 gate possible at all -- and it
# is direct evidence for replacing the finite difference with a closed form.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${1:-$repo/build}"

# $1 = output dir, $2 = "pair" to strip the DEP accumulation
build_variant() {
  local out="$1" mode="$2"
  mkdir -p "$out"
  cp "$repo"/legacy/fortran_bd/*.f "$repo"/legacy/fortran_bd/makefile "$out/"
  cp "$repo"/data/run.txt "$repo"/data/start.txt "$repo"/data/2dtabledssnp300.txt "$out/"

  # Dump immediately before the time loop: after readcn and after every COMMON
  # constant is set, so forces.f runs with exactly the real setup. No RNG is
  # involved -- forces() is a pure function of (positions, lambda, config).
  awk '
    /^      do l=1, nstep/ && !done {
      print "c    ** FORCE-DUMP INSTRUMENTATION (bd_csa regression harness) **"
      print "      call forces(F, r, nxyz)"
      print "      open(77,file=\x27force_dump.txt\x27)"
      print "      write(77,\x27(a,i8)\x27) \x27# np \x27, np"
      print "      write(77,\x27(a,es26.17e3)\x27) \x27# lambda \x27, lambda"
      print "      do i=1,np"
      print "        write(77,\x27(i6,5es26.17e3)\x27) i,"
      print "     +     r(nxyz(1,i)), r(nxyz(2,i)),"
      print "     +     F(nxyz(1,i)), F(nxyz(2,i)), F(nxyz(3,i))"
      print "      end do"
      print "      close(77)"
      print "      stop"
      done = 1
    }
    { print }
  ' "$repo/legacy/fortran_bd/main.f" > "$out/main.f"

  grep -q "FORCE-DUMP INSTRUMENTATION" "$out/main.f" ||
    { echo "error: time-loop anchor not found in main.f" >&2; exit 1; }

  if [[ "$mode" == "pair" ]]; then
    # Comment out the two DEP accumulation statements (forces.f:344,346).
    awk '
      /Fdepx\*\(radii\(i\)\/a\)\*\*3/ || /Fdepy\*\(radii\(i\)\/a\)\*\*3/ {
        print "c" substr($0, 2); next
      }
      { print }
    ' "$repo/legacy/fortran_bd/forces.f" > "$out/forces.f"

    grep -cq "^c.*Fdepx\*" "$out/forces.f" ||
      { echo "error: DEP accumulation not found in forces.f" >&2; exit 1; }
  fi

  make -C "$out" >/dev/null
  (cd "$out" && ./bdpd 30.0 -7 >/dev/null)
  [[ -s "$out/force_dump.txt" ]] ||
    { echo "error: $out/force_dump.txt not written" >&2; exit 1; }
  echo "  $out/force_dump.txt"
}

echo "building force oracles:"
build_variant "$root/oracle_forcedump" full
build_variant "$root/oracle_forcedump_pair" pair
