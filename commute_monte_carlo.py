"""
Commute Monte Carlo Simulator
Run with:  pip install streamlit plotly numpy pandas
           streamlit run commute_monte_carlo.py
"""
from dataclasses import dataclass, field, replace
import numpy as np

TRAFFIC = ["Light", "Moderate", "Heavy", "Severe"]


# ----------------------------------------------------------------- engine
@dataclass
class Params:
    depart: int = 6 * 60 + 40          # minutes after midnight
    required: int = 7 * 60 + 30
    walk_to: tuple = (5, 8, 12)
    walk_from: tuple = (3, 5, 10)
    wait: tuple = (2, 7, 20)
    load: tuple = (1, 3, 5)
    transfer_on: bool = False
    transfer: tuple = (5, 8, 15)
    traffic_p: tuple = (30, 50, 17, 3)  # percent
    traffic_rng: tuple = ((20, 25), (25, 35), (35, 50), (50, 80))
    delay_p: float = 10.0               # percent
    delay_rng: tuple = (5, 30)


def tri(rng, t, n):
    lo, mode, hi = t
    if hi <= lo:
        return np.full(n, float(lo))
    return rng.triangular(lo, min(max(mode, lo), hi), hi, n)


def simulate(p: Params, n: int, seed=None):
    """Vectorised Monte Carlo: returns durations (min) and traffic indices."""
    rng = np.random.default_rng(seed)
    probs = np.array(p.traffic_p, float) / 100
    traffic = rng.choice(4, size=n, p=probs / probs.sum())      # weighted pick
    lo = np.array([r[0] for r in p.traffic_rng], float)[traffic]
    hi = np.array([r[1] for r in p.traffic_rng], float)[traffic]
    travel = rng.uniform(lo, hi)
    delay = np.where(rng.random(n) < p.delay_p / 100,           # Bernoulli
                     rng.uniform(*p.delay_rng, n), 0.0)
    dur = (tri(rng, p.walk_to, n) + tri(rng, p.wait, n) + travel
           + tri(rng, p.load, n) + delay + tri(rng, p.walk_from, n))
    if p.transfer_on:
        dur = dur + tri(rng, p.transfer, n)
    return dur, traffic


def summarize(p: Params, dur, traffic):
    arrive = p.depart + dur
    slack = p.required - arrive                                  # >0 = early
    pct = np.percentile(arrive, [5, 50, 95])
    return dict(
        dur=dur, arrive=arrive, traffic=traffic,
        p_on=float(np.mean(slack >= 0)), p_late=float(np.mean(slack < 0)),
        p_early10=float(np.mean(slack >= 10)),
        mean_dur=float(dur.mean()), med_dur=float(np.median(dur)),
        sd_dur=float(dur.std(ddof=1)), mean_arr=float(arrive.mean()),
        min_arr=float(arrive.min()), max_arr=float(arrive.max()),
        min_dur=float(dur.min()), max_dur=float(dur.max()),
        p5=pct[0], p50=pct[1], p95=pct[2],
        n_early=int(np.sum(slack >= 10)),
        n_ontime=int(np.sum((slack >= 0) & (slack < 10))),
        n_late=int(np.sum(slack < 0)),
    )


def safest_departure(p: Params, dur, target):
    """Latest departure (whole minute) with P(on time) >= target.
    Duration doesn't depend on departure, so one simulation covers all departures."""
    q = np.quantile(dur, target)
    dep = int(np.floor(p.required - q))
    return dep, float(np.mean(dep + dur <= p.required))


def prob_curve(p: Params, dur, span=60):
    base = int(p.required - np.quantile(dur, 0.5))
    xs = np.arange(base - span, base + span // 2 + 1)
    return xs, np.array([np.mean(x + dur <= p.required) for x in xs]) * 100


def scenarios(p: Params):
    tp = np.array(p.traffic_p, float)
    tp[2:] *= 0.5                                   # halve heavy/severe
    tp[:2] += (100 - tp.sum()) * tp[:2] / tp[:2].sum()
    return {
        "Current": p,
        "Leave Earlier (10 min)": replace(p, depart=p.depart - 10),
        "Avoid Traffic": replace(p, traffic_p=tuple(tp)),
        "Faster Route": replace(p, traffic_rng=tuple((a * .85, b * .85) for a, b in p.traffic_rng)),
    }


def fmt(m):
    m = int(round(m)) % (24 * 60)
    h, mi = divmod(m, 60)
    return f"{(h % 12) or 12}:{mi:02d} {'AM' if h < 12 else 'PM'}"


def interpret(p, s, n):
    txt = (f"Based on {n:,} simulated commutes, you have a **{s['p_on']*100:.1f}%** probability of "
           f"arriving by {fmt(p.required)}. Your average commute is about {s['mean_dur']:.1f} minutes. "
           f"Half of simulated trips arrive by {fmt(s['p50'])}, and 90% arrive between "
           f"{fmt(s['p5'])} and {fmt(s['p95'])}.")
    if s["p_late"] > 0.15:
        txt += (" Your current departure time carries a relatively high risk of lateness. "
                "Consider leaving earlier or reducing waiting and transfer times.")
    elif s["p_late"] < 0.05:
        txt += " Your current plan looks very reliable."
    return txt


# --------------------------------------------------------------------- UI
def main():
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd

    st.set_page_config("Commute Monte Carlo Simulator", "🚌", layout="wide")
    st.markdown("""<style>
    .block-container{max-width:1200px;padding-top:2rem}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:18px 20px;
          box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:12px}
    .card .l{color:#475569;font-size:.85rem}.card .v{font-size:1.8rem;font-weight:700;color:#0f172a}
    .hero{background:linear-gradient(135deg,#1e3a8a,#2563eb);border-radius:20px;padding:26px;color:#fff;
          box-shadow:0 4px 14px rgba(37,99,235,.35);margin-bottom:12px}
    .hero .v{font-size:3.4rem;font-weight:800;line-height:1.1}.hero .l{opacity:.9}
    </style>""", unsafe_allow_html=True)

    st.title("Commute Monte Carlo Simulator")
    st.markdown("*How likely are you to arrive on time?*")
    st.info("Monte Carlo simulation generates thousands of possible commuting scenarios using uncertain "
            "variables such as traffic, waiting time, travel time, and unexpected delays. The results "
            "estimate the probability of different arrival outcomes.")

    def to_min(t): return t.hour * 60 + t.minute

    def tri_inputs(label, d, key):
        st.caption(label)
        c = st.columns(3)
        return (c[0].number_input("Min", 0.0, 300.0, float(d[0]), 1.0, key=key + "a"),
                c[1].number_input("Most likely", 0.0, 300.0, float(d[1]), 1.0, key=key + "b"),
                c[2].number_input("Max", 0.0, 300.0, float(d[2]), 1.0, key=key + "c"))

    import datetime as dt
    with st.sidebar:
        st.header("Inputs")
        st.subheader("Schedule")
        dep = st.time_input("Departure time", dt.time(6, 40))
        req = st.time_input("Required arrival time", dt.time(7, 30))
        st.subheader("Walking")
        walk_to = tri_inputs("To transportation stop (min)", (5, 8, 12), "wt")
        walk_from = tri_inputs("From final stop to destination (min)", (3, 5, 10), "wf")
        st.subheader("Waiting")
        wait = tri_inputs("Waiting time (min)", (2, 7, 20), "wa")
        st.subheader("Traffic")
        defaults_p, defaults_r = (30, 50, 17, 3), ((20, 25), (25, 35), (35, 50), (50, 80))
        tp, tr = [], []
        for i, name in enumerate(TRAFFIC):
            c = st.columns(3)
            tp.append(c[0].number_input(f"{name} %", 0.0, 100.0, float(defaults_p[i]), 1.0, key=f"tp{i}"))
            tr.append((c[1].number_input("Min", 0.0, 300.0, float(defaults_r[i][0]), key=f"tl{i}"),
                       c[2].number_input("Max", 0.0, 300.0, float(defaults_r[i][1]), key=f"th{i}")))
        total = sum(tp)
        if abs(total - 100) > 1e-6:
            st.error(f"Traffic probabilities must add up to 100% (currently {total:g}%).")
        st.subheader("Loading / unloading")
        load = tri_inputs("Loading time (min)", (1, 3, 5), "ld")
        st.subheader("Transfer")
        tron = st.toggle("Do you need to transfer vehicles?", False)
        trf = tri_inputs("Transfer time (min)", (5, 8, 15), "tf") if tron else (5, 8, 15)
        st.subheader("Unexpected delay")
        dp = st.slider("Probability of delay (%)", 0, 100, 10)
        c = st.columns(2)
        dmin = c[0].number_input("Min delay", 0.0, 300.0, 5.0, key="dmin")
        dmax = c[1].number_input("Max delay", 0.0, 300.0, 30.0, key="dmax")
        st.subheader("Simulation")
        n = st.selectbox("Number of simulations", [1000, 10000, 50000, 100000], index=1)
        run = st.button("Run Simulation", type="primary", use_container_width=True)

    for nm, t in [("walking to stop", walk_to), ("walking from stop", walk_from), ("waiting", wait),
                  ("loading", load), ("transfer", trf)]:
        if not (t[0] <= t[1] <= t[2]):
            st.error(f"For {nm}, values must satisfy min ≤ most likely ≤ max.")
            st.stop()
    if abs(total - 100) > 1e-6:
        st.stop()

    p = Params(to_min(dep), to_min(req), walk_to, walk_from, wait, load, tron, trf,
               tuple(tp), tuple(tr), float(dp), (dmin, dmax))
    if p.required <= p.depart:
        st.error("Required arrival time must be after departure time.")
        st.stop()

    if run or "res" not in st.session_state:
        with st.spinner("Simulating..."):
            dur, traffic = simulate(p, n)
            st.session_state.res = dict(p=p, n=n, dur=dur, traffic=traffic)
    R = st.session_state.res
    p, n, dur, traffic = R["p"], R["n"], R["dur"], R["traffic"]
    s = summarize(p, dur, traffic)

    # ---- summary cards
    top = st.columns([1.6, 1, 1])
    top[0].markdown(f"<div class='hero'><div class='l'>Probability of arriving by {fmt(p.required)}</div>"
                    f"<div class='v'>{s['p_on']*100:.1f}%</div>"
                    f"<div class='l'>{s['p_early10']*100:.1f}% arrive at least 10 min early</div></div>",
                    unsafe_allow_html=True)

    def card(col, label, val):
        col.markdown(f"<div class='card'><div class='l'>{label}</div><div class='v'>{val}</div></div>",
                     unsafe_allow_html=True)
    card(top[1], "Probability of being late", f"{s['p_late']*100:.1f}%")
    card(top[2], "Average commute", f"{s['mean_dur']:.1f} min")
    row = st.columns(3)
    card(row[0], "Expected arrival", fmt(s["mean_arr"]))
    card(row[1], "Earliest simulated arrival", fmt(s["min_arr"]))
    card(row[2], "Latest simulated arrival", fmt(s["max_arr"]))

    st.subheader("What does the simulation tell you?")
    st.success(interpret(p, s, n))

    # ---- charts
    c1, c2 = st.columns(2)
    edges = np.arange(np.floor(s["min_arr"]), np.ceil(s["max_arr"]) + 2, 1.0)
    cnt, _ = np.histogram(s["arrive"], bins=edges)
    ctr = edges[:-1] + .5
    f1 = go.Figure()
    for mask, name, col in [(ctr <= p.required, "On time / early", "#16a34a"),
                            (ctr > p.required, "Late", "#dc2626")]:
        f1.add_bar(x=ctr[mask], y=cnt[mask], name=name, marker_color=col)
    f1.add_vline(x=p.required, line_dash="dash", line_color="#0f172a",
                 annotation_text=f"Deadline {fmt(p.required)}")
    ticks = np.arange(np.ceil(edges[0] / 10) * 10, edges[-1], 10)
    f1.update_layout(title="Arrival time distribution", xaxis_title="Arrival time",
                     yaxis_title="Number of simulations", bargap=0.02, height=380,
                     xaxis=dict(tickvals=ticks, ticktext=[fmt(t) for t in ticks]))
    c1.plotly_chart(f1, use_container_width=True)

    f2 = go.Figure(go.Histogram(x=dur, xbins=dict(size=1), marker_color="#2563eb"))
    f2.add_vline(x=s["mean_dur"], line_dash="dash", annotation_text=f"Mean {s['mean_dur']:.1f}")
    f2.update_layout(title="Commute duration distribution", xaxis_title="Duration (minutes)",
                     yaxis_title="Number of simulations", bargap=0.02, height=380)
    c2.plotly_chart(f2, use_container_width=True)

    c3, c4 = st.columns(2)
    sim_pct = np.bincount(traffic, minlength=4) / n * 100
    f3 = go.Figure(go.Pie(labels=TRAFFIC, values=sim_pct, hole=.55, sort=False,
                          marker_colors=["#22c55e", "#eab308", "#f97316", "#dc2626"],
                          customdata=p.traffic_p,
                          hovertemplate="%{label}<br>Simulated %{value:.1f}%<br>Configured %{customdata}%<extra></extra>"))
    f3.update_layout(title="Traffic conditions (simulated)", height=380)
    c3.plotly_chart(f3, use_container_width=True)
    c3.caption("Configured vs simulated: " + " · ".join(
        f"{t} {cfg:g}% → {sm:.1f}%" for t, cfg, sm in zip(TRAFFIC, p.traffic_p, sim_pct)))

    f4 = go.Figure(go.Bar(x=["Early (≥10 min)", "On time (<10 min early)", "Late"],
                          y=[s["n_early"], s["n_ontime"], s["n_late"]],
                          marker_color=["#16a34a", "#2563eb", "#dc2626"]))
    f4.update_layout(title="Outcome distribution", yaxis_title="Number of simulations", height=380)
    c4.plotly_chart(f4, use_container_width=True)

    # ---- stats table
    with st.expander("Detailed statistics"):
        st.table(pd.DataFrame({
            "Statistic": ["Mean duration", "Median duration", "Std. deviation", "Min duration", "Max duration",
                          "5th pct arrival", "50th pct arrival", "95th pct arrival"],
            "Value": [f"{s['mean_dur']:.2f} min", f"{s['med_dur']:.2f} min", f"{s['sd_dur']:.2f} min",
                      f"{s['min_dur']:.1f} min", f"{s['max_dur']:.1f} min",
                      fmt(s["p5"]), fmt(s["p50"]), fmt(s["p95"])]}))

    # ---- optimal departure
    st.header("Find My Safest Departure Time")
    tgt = st.select_slider("Target reliability", [80, 85, 90, 95, 99], value=95, format_func=lambda x: f"{x}%")
    best, achieved = safest_departure(p, dur, tgt / 100)
    st.success(f"**Recommended departure: {fmt(best)}** — estimated **{achieved*100:.1f}%** probability "
               f"of arriving by {fmt(p.required)}. That is {p.depart - best:+d} min relative to your "
               f"current departure (negative means you could leave later).")
    xs, ys = prob_curve(p, dur)
    f5 = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#2563eb", width=3)))
    f5.add_hline(y=tgt, line_dash="dot", line_color="#64748b")
    f5.add_vline(x=best, line_dash="dash", line_color="#16a34a", annotation_text="Recommended")
    f5.update_layout(title="Departure time vs. probability of arriving on time", height=360,
                     yaxis_title="P(on time) %", xaxis_title="Departure time",
                     xaxis=dict(tickvals=xs[::10], ticktext=[fmt(x) for x in xs[::10]]))
    st.plotly_chart(f5, use_container_width=True)

    # ---- scenario comparison
    st.header("Scenario Comparison")
    rows = []
    for name, sp in scenarios(p).items():
        d, t = simulate(sp, n)
        ss = summarize(sp, d, t)
        rows.append({"Scenario": name, "On-Time Probability": f"{ss['p_on']*100:.1f}%",
                     "Avg. Commute": f"{ss['mean_dur']:.1f} min", "Avg. Arrival": fmt(ss["mean_arr"])})
    st.table(pd.DataFrame(rows).set_index("Scenario"))
    st.caption("Avoid Traffic halves heavy/severe probabilities; Faster Route cuts travel-time ranges by 15%.")

    # ---- education
    st.header("How Monte Carlo Simulation Works")
    st.markdown("**Uncertain Variables** → **Randomly Generate One Commute** → **Calculate Arrival Time** → "
                "**Repeat Thousands of Times** → **Analyze the Results** → **Estimate Probability**")
    st.write("Instead of trying to predict exactly what will happen tomorrow, Monte Carlo simulation creates "
             "thousands of plausible versions of tomorrow. The percentage of simulations producing a "
             "particular outcome becomes an estimate of that outcome's probability.")
    st.warning("**Simulation results are estimates, not guarantees.** The accuracy of the prediction depends "
               "on how realistic the input distributions and probabilities are. Real-world conditions can "
               "produce outcomes outside the simulated assumptions.")


if __name__ == "__main__":
    main()