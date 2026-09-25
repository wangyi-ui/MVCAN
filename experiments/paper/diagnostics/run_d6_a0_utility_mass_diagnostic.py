from .d6_a0_utility_mass_diagnostic import run

if __name__ == "__main__":
    report = run()
    print("[D6-A0] DATASET UTILITY MASS", flush=True)
    for row in report["rows"]:
        print("%s seed%s rho_global=%.12f rho_batch_mean=%.12f min=%.12f max=%.12f" % (row["dataset"], row["seed"], row["rho_global"], row["rho_batch"]["mean"], row["rho_batch"]["min"], row["rho_batch"]["max"]), flush=True)
    for gate, value in report["preregistered_gate"].items():
        print("[D6-A0] %s = %s" % (gate.upper(), "PASS" if value else "FAIL"), flush=True)
    print("[D6-A0] FINAL DECISION = " + report["final_decision"], flush=True)
