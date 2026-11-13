"""ArecaGrow AI - Arecanut Growth Prediction & Analytics Dashboard.  Run:  streamlit run app.py"""
import io, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
import statsmodels.api as sm
from src.config import *
from src.preprocessing import clean_dataset, normalise_columns, missing_required
from src.feature_engineering import prepare_model_frame
from src.pipeline import load_default_raw, run_training
from src.health import fit_reference, load_reference, score_frame, score_row, explain, TRAITS
from src.prediction import (models_available, load_bundles, input_frame, predict_interval, project_growth,
                            predict_yield, batch_predict, report_markdown)
from src import explainability as xai

st.set_page_config(page_title="ArecaGrow AI", page_icon="🌴", layout="wide")
GREEN, LIME, BROWN, YELLOW, CREAM = "#1B5E20", "#7CB518", "#8B5E3C", "#E0B84C", "#FAF7EC"
PALETTE = [GREEN, LIME, BROWN, YELLOW, "#4C956C", "#A3B18A"]
st.markdown(f"""<style>
.stApp{{background:{CREAM}}} [data-testid=stSidebar]{{background:{GREEN}}}
[data-testid=stSidebar] *{{color:#F4F9E9 !important}}
h1,h2,h3{{color:{GREEN}; font-family:Georgia,serif}}
.kpi{{background:#fff;border-left:6px solid {LIME};border-radius:10px;padding:14px 16px;box-shadow:0 1px 4px #0002}}
.kpi .v{{font-size:1.7rem;font-weight:700;color:{GREEN}}} .kpi .l{{color:{BROWN};font-size:.85rem}}
.tag-pred{{background:{LIME};color:#fff;border-radius:6px;padding:2px 8px;font-size:.75rem;font-weight:600}}
.tag-meas{{background:{BROWN};color:#fff;border-radius:6px;padding:2px 8px;font-size:.75rem;font-weight:600}}
.banner{{background:#FFF3CD;border:1px solid {YELLOW};border-radius:8px;padding:8px 12px;color:#5c4a00}}
</style>""", unsafe_allow_html=True)


def kpi(col, icon, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{icon} {label}</div><div class="v">{value}</div>'
                 f'<div class="l">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff", colorway=PALETTE,
                      margin=dict(l=10, r=10, t=45, b=10), font=dict(family="Georgia"))
    return fig


# ------------------------------------------------------------------ data & model state
@st.cache_data(show_spinner=False)
def _prep(raw: pd.DataFrame):
    clean, rep = clean_dataset(raw); return clean, rep, prepare_model_frame(clean)


@st.cache_resource(show_spinner=False)
def _bundles(stamp): return load_bundles()


if "raw" not in st.session_state:
    st.session_state.raw, st.session_state.source = load_default_raw(), "SYNTHETIC (simulated) - data/arecanut_synthetic.csv"
if not models_available():
    with st.spinner("First run: training models on the dataset (about 1-3 minutes)..."):
        run_training(st.session_state.raw, st.session_state.source)
    st.cache_resource.clear()
clean, report, mf = _prep(st.session_state.raw)
B = _bundles(str(max(p.stat().st_mtime for p in MODELS_DIR.glob("*.joblib"))))
ref = load_reference() or fit_reference(clean)
is_synth = "SYNTHETIC" in st.session_state.source.upper()
latest = clean[clean[YEAR] == clean[YEAR].max()].copy()
scored = score_frame(clean, ref); scored_latest = scored[scored[YEAR] == scored[YEAR].max()]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌴 ArecaGrow AI")
PAGES = ["Overview", "Data Analysis & Upload", "Growth Prediction", "Yield Prediction", "Soil & Weather Impact",
         "Plantation Health", "Explainable AI", "Model Performance", "Interactive Prediction", "Batch Prediction & Reports"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption(f"**Dataset:** {st.session_state.source}\n\n{len(clean):,} plantation-year records · {clean[ID].nunique()} plantations")
if is_synth: st.sidebar.warning("Synthetic data: results show the method works, not real agronomy.")
if not xai.HAS_SHAP: st.sidebar.info("`shap` not installed - using permutation importance fallback.")
st.title(page)
if is_synth:
    st.markdown('<div class="banner">⚠️ <b>SYNTHETIC DATA.</b> All observations shown are simulated for development and testing. '
                'They are not real farm measurements. Upload verified data on the <i>Data Analysis &amp; Upload</i> page.</div>', unsafe_allow_html=True)
st.write("")

# ================================================================== A. OVERVIEW
if page == "Overview":
    pred = batch_predict(B, mf[mf[YEAR] == mf[YEAR].max()].copy())
    c = st.columns(6)
    kpi(c[0], "🌳", "Plantations analysed", f"{clean[ID].nunique()}", f"latest year {int(clean[YEAR].max())}")
    kpi(c[1], "📅", "Average age", f"{latest.plantation_age.mean():.1f} yr", "observed")
    kpi(c[2], "📈", "Avg height growth", f"{mf.growth_height_m.mean()*100:.0f} cm/yr", "observed, year-on-year")
    kpi(c[3], "🥥", "Predicted yield", f"{pred.pred_yield_per_ha_kg.mean():,.0f} kg/ha", "model estimate")
    kpi(c[4], "💧", "Avg soil moisture", f"{latest.soil_moisture.mean():.1f} %", "observed")
    kpi(c[5], "❤️", "Health score", f"{scored_latest.health_score.mean():.0f}/100", "relative index")
    st.write("")
    a, b = st.columns(2)
    yr = clean.groupby(YEAR).agg(height=("tree_height_m", "mean"), girth=("stem_girth_cm", "mean"), yld=("yield_per_tree_kg", "mean")).reset_index()
    f = go.Figure(); f.add_scatter(x=yr[YEAR], y=yr.height, name="Mean height (m)", line=dict(color=GREEN))
    f.add_scatter(x=yr[YEAR], y=yr.girth, name="Mean girth (cm)", yaxis="y2", line=dict(color=BROWN))
    f.update_layout(title="Growth trend (observed)", yaxis2=dict(overlaying="y", side="right")); a.plotly_chart(style(f))
    loc = clean.groupby("location").yield_per_tree_kg.mean().reset_index()
    b.plotly_chart(style(px.bar(loc, x="location", y="yield_per_tree_kg", title="Mean observed yield per tree by location (kg/yr)")))
    a, b = st.columns(2)
    gd = mf.dropna(subset=["growth_height_m"])
    a.plotly_chart(style(px.scatter(gd, x="rainfall", y="growth_height_m", color="soil_type", opacity=.5, trendline="ols", trendline_scope="overall",
                    title="Rainfall vs annual height growth (observed)", labels=dict(growth_height_m="height growth (m/yr)"))))
    summ = scored_latest.groupby("location").agg(plantations=(ID, "nunique"), mean_age=("plantation_age", "mean"),
            mean_yield_kg_tree=("yield_per_tree_kg", "mean"), health=("health_score", "mean")).round(2).reset_index()
    b.subheader("Plantation performance summary"); b.dataframe(summ, hide_index=True)

# ================================================================== B. DATA ANALYSIS
elif page == "Data Analysis & Upload":
    up = st.file_uploader("Upload plantation CSV", type="csv")
    st.caption("Required columns: " + ", ".join(REQUIRED_COLS) + ".  Several common aliases are accepted (e.g. `fertilizer_quantity`).")
    st.download_button("Download CSV template", pd.DataFrame(columns=REQUIRED_COLS).to_csv(index=False), "arecagrow_template.csv")
    if up is not None:
        new = normalise_columns(pd.read_csv(up)); miss = missing_required(new)
        if miss: st.error(f"Missing columns: {miss}")
        else:
            st.success(f"{len(new)} rows read.")
            if st.button("Use this dataset and retrain all models", type="primary"):
                with st.spinner("Cleaning and retraining..."):
                    st.session_state.raw, st.session_state.source = new, f"UPLOADED: {up.name}"
                    run_training(new, f"UPLOADED: {up.name}")
                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    if st.button("Reset to built-in synthetic dataset"):
        st.session_state.pop("raw"); st.cache_data.clear(); st.rerun()
    t1, t2, t3, t4 = st.tabs(["Cleaning report", "Feature engineering", "EDA", "Units & sources"])
    with t1:
        st.json(report); st.caption("Duplicates removed, impossible values blanked, slow-changing traits interpolated within each plantation, "
                                    "remaining gaps median/mode-imputed, outliers flagged (1.5×IQR) and only extreme unbounded values capped (3×IQR).")
        st.dataframe(clean.head(50))
    with t2:
        st.write("Engineered features: " + ", ".join(ENGINEERED)); st.write("Targets: next-year **increments** (`growth_*`) and `prev_yield_per_tree_kg` lag.")
        st.dataframe(mf[[ID, YEAR] + ENGINEERED + list(GROWTH_TARGETS) + ["prev_yield_per_tree_kg"]].head(50))
    with t3:
        st.dataframe(clean.describe().T.round(2))
        col = st.selectbox("Distribution of", RAW_NUMERIC[1:], index=5)
        st.plotly_chart(style(px.histogram(clean, x=col, color="soil_type", marginal="box", nbins=40)))
        num = clean.select_dtypes("number").drop(columns=["outlier_flag"], errors="ignore")
        st.plotly_chart(style(px.imshow(num.corr().round(2), text_auto=".1f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1, title="Correlation matrix"), 650))
    with t4:
        st.table(pd.DataFrame({"column": UNITS.keys(), "unit": UNITS.values()}))
        st.markdown("**Sources:** the bundled dataset is *synthetic*. For real work use verified data (ICAR-CPCRI, Kerala Agricultural University, "
                    "UAS/Karnataka institutions, published studies) and record the collection period, units and licence here. "
                    "**Limitations:** simulated relationships; no disease labels; weather is annual/monthly mean, not daily.")

# ================================================================== C. GROWTH PREDICTION
elif page == "Growth Prediction":
    tgt = st.selectbox("Growth target", list(GROWTH_TARGETS), format_func=GROWTH_TARGETS.get); b = B[tgt]
    cv = b["metrics"][b["metrics"].split.str.startswith("CV")].sort_values("RMSE")
    st.success(f"Best model (lowest cross-validated RMSE): **{b['best_model']}**")
    st.dataframe(cv[["model", "MAE", "MSE", "RMSE", "R2"]].round(4), hide_index=True)
    st.subheader("Project a plantation forward")
    c = st.columns(3); pid = c[0].selectbox("Plantation", sorted(latest[ID])); yrs = c[1].slider("Years ahead", 1, 10, 3)
    r = latest[latest[ID] == pid].iloc[0].to_dict(); r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
    proj = project_growth(B, r, yrs)
    X = input_frame(r)
    for t, lab in GROWTH_TARGETS.items():
        p, lo, hi = predict_interval(B[t], X); c[2].metric(f"Next-year {lab}", f"{p[0]:.2f}", f"90% band {lo[0]:.2f} to {hi[0]:.2f}", delta_color="off")
    f = go.Figure()
    for col, nm in [("height_m", "Height (m)"), ("girth_cm", "Girth (cm)")]:
        f.add_scatter(x=proj.age, y=proj[col], name=nm, mode="lines+markers", yaxis="y2" if col == "girth_cm" else "y")
    f.update_layout(title="Entered state (age 0 point) then PREDICTED trajectory", xaxis_title="Plantation age (yr)", yaxis2=dict(overlaying="y", side="right"))
    st.plotly_chart(style(f)); st.dataframe(proj.round(2), hide_index=True)
    st.caption("Recursive projection assuming conditions stay constant; uncertainty widens with horizon. The first row is measured, later rows are predicted.")

# ================================================================== D. YIELD PREDICTION
elif page == "Yield Prediction":
    yb = B[YIELD_TARGET]; st.success(f"Best yield model: **{yb['best_model']}**")
    c = st.columns(2); pid = c[0].selectbox("Plantation", sorted(mf[ID].unique()))
    hist = mf[(mf[ID] == pid)].dropna(subset=["prev_yield_per_tree_kg"]); yr = c[1].selectbox("Year", sorted(hist[YEAR], reverse=True))
    r = hist[hist[YEAR] == yr].iloc[0].to_dict(); y = predict_yield(B, r)
    k = st.columns(4)
    kpi(k[0], "🌴", "Predicted yield / tree", f"{y['per_tree_kg']:.2f} kg", f"90% band {y['lo']:.2f}-{y['hi']:.2f}")
    kpi(k[1], "📦", "Total predicted yield", f"{y['total_kg']:,.0f} kg", f"{y['productive_trees']:,} bearing palms")
    kpi(k[2], "🌾", "Predicted yield / ha", f"{y['per_ha_kg']:,.0f} kg", "model estimate")
    kpi(k[3], "🧾", "Observed yield / tree", f"{r['yield_per_tree_kg']:.2f} kg", "measured record")
    h = mf[mf[ID] == pid].sort_values(YEAR); hp = batch_predict(B, h.copy())
    f = go.Figure(); f.add_scatter(x=h[YEAR], y=h.yield_per_tree_kg, name="Observed (historical)", line=dict(color=BROWN))
    f.add_scatter(x=hp[YEAR], y=hp.pred_yield_per_tree_kg, name="Model-predicted", line=dict(color=LIME, dash="dash"))
    f.update_layout(title=f"{pid}: observed vs predicted yield per tree (kg/yr)"); st.plotly_chart(style(f))
    st.caption("Predictions use the previous year's recorded yield as one input. In-sample years for a training plantation look better than genuinely new plantations - see Model Performance for honest hold-out scores.")

# ================================================================== E. IMPACT ANALYSIS
elif page == "Soil & Weather Impact":
    pairs = [("rainfall", "growth_height_m"), ("soil_moisture", "growth_height_m"), ("temperature", "growth_height_m"),
             ("fertilizer_kg_ha", "yield_per_tree_kg"), ("plantation_age", "yield_per_tree_kg")]
    cols = st.columns(2)
    for i, (x, yv) in enumerate(pairs):
        d = mf.dropna(subset=[yv]); d = d[d.plantation_age >= PRODUCTIVE_AGE] if yv == "yield_per_tree_kg" and x != "plantation_age" else d
        cols[i % 2].plotly_chart(style(px.scatter(d, x=x, y=yv, color="soil_type", opacity=.55, trendline="ols", trendline_scope="overall",
                                 title=f"{x} vs {yv}" + (" (bearing palms)" if yv == "yield_per_tree_kg" and x != "plantation_age" else ""))))
    st.subheader("Most influential factors")
    tab = st.selectbox("Outcome", ["growth_height_m", "yield_per_tree_kg"]); fe = GROWTH_FEATURES if tab.startswith("g") else YIELD_FEATURES
    bundle = B[tab] if tab != "yield_per_tree_kg" else B[YIELD_TARGET]
    d = mf.dropna(subset=[tab] + [f for f in fe if f != "prev_yield_per_tree_kg"] ); d = d.dropna(subset=["prev_yield_per_tree_kg"]) if "prev_yield_per_tree_kg" in fe else d
    imp, how = xai.global_importance(bundle, d.sample(min(500, len(d)), random_state=0), d.loc[d.sample(min(500, len(d)), random_state=0).index, tab])
    st.plotly_chart(style(px.bar(imp.head(12).iloc[::-1], orientation="h", title=f"Model-based importance ({how})")))
    num = [f for f in fe if f not in CATEGORICAL]; Z = (d[num] - d[num].mean()) / d[num].std()
    ols = sm.OLS((d[tab] - d[tab].mean()) / d[tab].std(), sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": d[ID].astype("category").cat.codes})
    st.write(f"**Statsmodels OLS (standardised, cluster-robust by plantation)** - R² = {ols.rsquared:.2f}")
    st.dataframe(pd.DataFrame({"std_coef": ols.params, "p_value": ols.pvalues}).drop("const").sort_values("std_coef", key=abs, ascending=False).round(4))
    st.caption("Associations, not proof of causation - especially with correlated inputs.")
    st.subheader("Time series (statsmodels Holt smoothing)")
    ys = clean.groupby(YEAR).yield_per_tree_kg.mean()
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(ys.values, trend="add").fit(); fc = fit.forecast(3)
    f = go.Figure(); f.add_scatter(x=ys.index, y=ys.values, name="Observed annual mean")
    f.add_scatter(x=list(range(ys.index[-1] + 1, ys.index[-1] + 4)), y=fc, name="Forecast", line=dict(dash="dash"))
    st.plotly_chart(style(f.update_layout(title="Mean yield per tree: trend forecast (naive, short series)")))

# ================================================================== F. HEALTH
elif page == "Plantation Health":
    st.info("The health score is a **relative growth-performance index** (height, girth, leaf count and, for bearing palms, yield versus age-matched peers). "
            "It does **not** diagnose disease - no disease-labelled data is available.")
    c = st.columns(3)
    counts = scored_latest.growth_category.value_counts().reindex(["Healthy growth", "Moderate growth", "Poor growth"]).fillna(0)
    c[0].plotly_chart(style(px.pie(values=counts.values, names=counts.index, hole=.5, color=counts.index, title="Latest-year classification",
                                   color_discrete_map={"Healthy growth": LIME, "Moderate growth": YELLOW, "Poor growth": BROWN}), 330))
    c[1].plotly_chart(style(px.histogram(scored_latest, x="health_score", nbins=25, title="Score distribution"), 330))
    c[2].plotly_chart(style(px.box(scored_latest, x="location", y="health_score", title="By location"), 330))
    pid = st.selectbox("Explain a plantation", sorted(scored_latest[ID])); r = scored_latest[scored_latest[ID] == pid].iloc[0].to_dict()
    sc, cat, comps = score_row(r, ref)
    st.subheader(f"{pid}: {cat} (score {sc:.0f}/100)")
    st.plotly_chart(style(px.bar(x=[TRAITS[k][0] for k in comps], y=list(comps.values()), labels=dict(x="", y="Z-score vs age peers"),
                    color=[v > 0 for v in comps.values()], color_discrete_map={True: LIME, False: BROWN}), 300))
    for line in explain(comps): st.write("• " + line)
    st.caption("Thresholds: ≥60 healthy, 40-60 moderate, <40 poor (score = 50 + 25 × weighted mean z-score).")
    st.dataframe(scored_latest[[ID, "location", "plantation_age", "health_score", "growth_category"]].round(1).sort_values("health_score"), hide_index=True)

# ================================================================== G. EXPLAINABLE AI
elif page == "Explainable AI":
    tgt = st.selectbox("Model", list(B), format_func=lambda t: B[t]["label"]); b = B[tgt]
    st.write(f"Explaining **{b['best_model']}** ({'SHAP' if xai.HAS_SHAP else 'permutation / ablation fallback - install shap for full SHAP plots'}).")
    Xs = b["X_ref"].sample(min(300, len(b["X_ref"])), random_state=0)
    imp, how = xai.global_importance(b, Xs, None if xai.HAS_SHAP else b["pipeline"].predict(Xs))  # fallback uses model output as target
    st.plotly_chart(style(px.bar(imp.head(12).iloc[::-1], orientation="h", title=f"Feature importance ({how})")))
    if xai.HAS_SHAP:
        st.pyplot(xai.summary_plot(b, Xs))
    st.subheader("Explain one prediction"); i = st.slider("Row from reference sample", 0, len(Xs) - 1, 0)
    row = Xs.iloc[[i]]; p, lo, hi = predict_interval(b, row)
    st.write(f"Predicted **{b['label']}**: **{p[0]:.2f}** (90% empirical band {lo[0]:.2f} to {hi[0]:.2f})")
    con = xai.local_contributions(b, row); top = con.head(10).iloc[::-1]
    st.plotly_chart(style(px.bar(x=top.values, y=top.index, orientation="h", color=top.values > 0, color_discrete_map={True: LIME, False: BROWN},
                    title="Contribution to this prediction (relative to average)", labels=dict(x="Effect on prediction", y="")), 380))
    a, c = st.columns(2)
    a.markdown("**⬆ Raising the prediction**"); [a.write(f"• {k} ({v:+.3f})") for k, v in con[con > 0].head(5).items()]
    c.markdown("**⬇ Lowering the prediction**"); [c.write(f"• {k} ({v:+.3f})") for k, v in con[con < 0].head(5).items()]
    st.caption("Explanations describe what the MODEL learned, not necessarily agronomic causation.")

# ================================================================== H. MODEL PERFORMANCE
elif page == "Model Performance":
    tgt = st.selectbox("Target", list(B), format_func=lambda t: B[t]["label"]); b = B[tgt]; m = b["metrics"]
    st.write(f"Trained on **{b['n_rows']:,} rows / {b['n_plantations']} plantations** · data: *{b['data_source']}* · "
             f"selected by CV RMSE: **{b['best_model']}**")
    split = st.radio("Validation scheme", list(m.split.unique()), horizontal=True)
    st.dataframe(m[m.split == split][["model", "R2", "MAE", "MSE", "RMSE"]].sort_values("RMSE").round(4), hide_index=True)
    st.caption("CV = GroupKFold by plantation (no plantation appears in both train and validation). Temporal hold-out trains on early years and tests on the last two. "
               "'Baseline (mean)' shows what no learning achieves. Scores on synthetic data only measure recovery of simulated rules.")
    ho = b["holdout"]; mod = st.selectbox("Model for plots", [c[6:] for c in ho if c.startswith("pred::")], index=[c[6:] for c in ho if c.startswith("pred::")].index(b["best_model"]))
    a, c = st.columns(2); pr = ho[f"pred::{mod}"]
    f = px.scatter(x=ho.actual, y=pr, labels=dict(x="Actual", y="Predicted"), title="Actual vs predicted (hold-out plantations)")
    lim = [min(ho.actual.min(), pr.min()), max(ho.actual.max(), pr.max())]; f.add_scatter(x=lim, y=lim, mode="lines", name="ideal", line=dict(color=BROWN, dash="dash"))
    a.plotly_chart(style(f)); c.plotly_chart(style(px.histogram(x=ho.actual - pr, nbins=30, title="Residuals (actual − predicted)", labels=dict(x="Residual"))))
    cf = b["cv_folds"]; cf = cf[cf.model != "Baseline (mean)"]
    st.plotly_chart(style(px.box(cf, x="model", y="R2", points="all", title="Cross-validation R² across 5 plantation-wise folds")))

# ================================================================== INTERACTIVE PREDICTION
elif page == "Interactive Prediction":
    st.write("Enter plantation details. Everything you type is **<span class='tag-meas'>ENTERED</span>**; model outputs are **<span class='tag-pred'>PREDICTED</span>**.", unsafe_allow_html=True)
    with st.form("f"):
        c = st.columns(4)
        age = c[0].number_input("Plantation age (yr)", 1, 60, 12); soil = c[1].selectbox("Soil type", SOIL_TYPES)
        ph = c[2].number_input("Soil pH", 4.0, 8.5, 6.0, .1); moist = c[3].number_input("Soil moisture (%)", 5.0, 60.0, 30.0)
        temp = c[0].number_input("Avg temperature (°C)", 15.0, 40.0, 27.0); hum = c[1].number_input("Humidity (%)", 30.0, 100.0, 80.0)
        rain = c[2].number_input("Monthly rainfall (mm)", 0.0, 800.0, 230.0); irr = c[3].number_input("Irrigations / month", 0, 15, 4)
        fert = c[0].number_input("Fertilizer (kg/ha/yr)", 0.0, 500.0, 150.0); ht = c[1].number_input("Tree height (m)", .3, 25.0, 8.5)
        gi = c[2].number_input("Stem girth (cm)", 5.0, 100.0, 40.0); lf = c[3].number_input("Leaf count", 1.0, 14.0, 8.0)
        c = st.columns(4); nt = c[0].number_input("Number of palms", 10, 20000, 1300); ar = c[1].number_input("Area (ha)", .1, 50.0, 1.0)
        pv = c[2].number_input("Previous year's yield (kg/palm)", 0.0, 8.0, 1.8); yrs = c[3].slider("Projection (years)", 1, 10, 3)
        go_ = st.form_submit_button("Predict Growth", type="primary")
    if go_:
        row = dict(plantation_age=age, soil_type=soil, soil_ph=ph, soil_moisture=moist, temperature=temp, humidity=hum, rainfall=rain,
                   irrigation_frequency=irr, fertilizer_kg_ha=fert, tree_height_m=ht, stem_girth_cm=gi, leaf_count=lf,
                   prev_yield_per_tree_kg=pv, num_trees=nt, area_ha=ar)
        proj = project_growth(B, row, yrs); y = predict_yield(B, row)
        sc, cat, comps = score_row(dict(row, yield_per_tree_kg=None), ref); X = input_frame(row)
        k = st.columns(4)
        kpi(k[0], "📈", "Height next year", f"{ht + predict_interval(B['growth_height_m'], X)[0][0]:.2f} m", "PREDICTED")
        kpi(k[1], "🥥", "Yield per tree", f"{y['per_tree_kg']:.2f} kg", f"PREDICTED · 90% band {y['lo']:.2f}-{y['hi']:.2f}")
        kpi(k[2], "🌾", "Yield per ha", f"{y['per_ha_kg']:,.0f} kg", f"total {y['total_kg']:,.0f} kg · PREDICTED")
        kpi(k[3], "❤️", "Growth category", cat, f"score {sc:.0f}/100 vs age peers · from ENTERED traits")
        st.write(""); st.subheader(f"Expected growth over {yrs} years"); st.dataframe(proj.round(2), hide_index=True)
        f = go.Figure(); f.add_scatter(x=proj.age, y=proj.height_m, mode="lines+markers", name="Height (m)", line=dict(color=GREEN))
        st.plotly_chart(style(f.update_layout(title="Height trajectory (first point entered, rest predicted)", xaxis_title="Age (yr)"), 300))
        st.subheader("What drives these predictions?")
        a, c = st.columns(2)
        for col, (t, nm) in zip([a, c], [("growth_height_m", "Height growth"), ("yield_per_tree_kg", "Yield per tree")]):
            con = xai.local_contributions(B[t], X); top = con.head(7).iloc[::-1]
            col.plotly_chart(style(px.bar(x=top.values, y=top.index, orientation="h", color=top.values > 0, color_discrete_map={True: LIME, False: BROWN},
                             title=f"{nm}: factor effects", labels=dict(x="", y="")), 330))
        st.caption(f"Uncertainty is an empirical 90% error band from plantation-wise cross-validation residuals; real-world error can be larger for plantations unlike the training data. Models trained on: {B[YIELD_TARGET]['data_source']}.")
        for line in explain(comps): st.write("• " + line)
        rep = report_markdown("ArecaGrow AI - Plantation Prediction Report", row, proj, y, cat, sc, B)
        st.download_button("⬇ Download prediction report (.md)", rep, "arecagrow_report.md")

# ================================================================== BATCH
else:
    st.write("Run all models on the currently loaded (cleaned) dataset - or upload a CSV on the *Data Analysis & Upload* page first.")
    sel = st.radio("Rows to predict", ["Latest year only", "All rows"], horizontal=True)
    d = mf[mf[YEAR] == mf[YEAR].max()] if sel.startswith("Latest") else mf
    out = batch_predict(B, d.copy())
    if out["prev_yield_imputed"].any(): st.warning(f"{int(out.prev_yield_imputed.sum())} rows had no previous-year yield; the training median was used (flagged in `prev_yield_imputed`).")
    show = [ID, YEAR, "location", "plantation_age", "yield_per_tree_kg", "pred_yield_per_tree_kg", "pred_yield_per_tree_kg_lo90", "pred_yield_per_tree_kg_hi90",
            "pred_yield_per_ha_kg", "pred_growth_height_m", "pred_growth_girth_cm", "pred_growth_leaf", "health_score", "growth_category"]
    st.dataframe(out[show].round(3), hide_index=True)
    st.caption("`yield_per_tree_kg` is the observed value; every `pred_*` column is a model estimate.")
    st.download_button("⬇ Download predictions (CSV)", out[show].to_csv(index=False), "arecagrow_predictions.csv")
    st.subheader("Summary report")
    s = out.groupby("location").agg(n=(ID, "nunique"), pred_yield_kg_tree=("pred_yield_per_tree_kg", "mean"), pred_yield_kg_ha=("pred_yield_per_ha_kg", "mean"),
                                    mean_health=("health_score", "mean")).round(2).reset_index()
    st.dataframe(s, hide_index=True)
    st.download_button("⬇ Download summary (CSV)", s.to_csv(index=False), "arecagrow_summary.csv")
