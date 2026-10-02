from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
	"Alcohol",
	"Malic_Acid",
	"Ash",
	"Ash_Alcanity",
	"Magnesium",
	"Total_Phenols",
	"Flavanoids",
	"Nonflavanoid_Phenols",
	"Proanthocyanins",
	"Color_Intensity",
	"Hue",
	"OD280",
	"Proline",
]
FEATURE_LABELS = {
	"Alcohol": "Alcohol",
	"Malic_Acid": "Malic acid",
	"Ash": "Ash",
	"Ash_Alcanity": "Ash alkalinity",
	"Magnesium": "Magnesium",
	"Total_Phenols": "Total phenols",
	"Flavanoids": "Flavanoids",
	"Nonflavanoid_Phenols": "Nonflavanoid phenols",
	"Proanthocyanins": "Proanthocyanins",
	"Color_Intensity": "Color intensity",
	"Hue": "Hue",
	"OD280": "OD280 / OD315",
	"Proline": "Proline",
}
DEFAULT_SAMPLE = {
	"Alcohol": 14.23,
	"Malic_Acid": 1.71,
	"Ash": 2.43,
	"Ash_Alcanity": 15.6,
	"Magnesium": 127,
	"Total_Phenols": 2.8,
	"Flavanoids": 3.06,
	"Nonflavanoid_Phenols": 0.28,
	"Proanthocyanins": 2.29,
	"Color_Intensity": 5.64,
	"Hue": 1.04,
	"OD280": 3.92,
	"Proline": 1065,
}
CLUSTER_COLORS = ["#b45d69", "#d39b51", "#518775"]
PAGES = ["Overview", "Predict a wine", "3D cluster map", "Cluster profiles", "Data studio"]

st.set_page_config(page_title="Wine Cluster Studio", page_icon="W", layout="wide")

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');
	:root {
		--wine-ink: #f3ecdf;
		--wine-muted: #c4b9a7;
		--wine-paper: #0c0e0d;
		--wine-panel: rgba(28, 29, 26, .92);
		--wine-line: rgba(230, 205, 158, .17);
		--wine-berry: #e3b44c;
		--wine-green: #b88424;
		--wine-copper: #d1a248;
	}
	html, body, .stApp, [data-testid="stAppViewContainer"] {
		background-color: var(--wine-paper) !important;
		color: var(--wine-ink) !important;
	}
	[data-testid="stAppViewContainer"] {
		background-image:
			repeating-radial-gradient(ellipse at 88% 8%, transparent 0 42px, rgba(223, 181, 92, .035) 43px 44px, transparent 45px 75px),
			linear-gradient(135deg, rgba(49, 42, 29, .38), transparent 48%, rgba(42, 31, 25, .24)) !important;
		background-attachment: fixed !important;
	}
	[data-testid="stHeader"] { background: rgba(10, 11, 10, .92) !important; }
	[data-testid="stToolbar"] { background: transparent !important; }
	.block-container { max-width: 1500px; padding-top: 2.25rem; padding-bottom: 3rem; }
	.stApp p, .stApp li, .stApp label, .stApp small,
	[data-testid="stMarkdownContainer"] p,
	[data-testid="stCaptionContainer"],
	[data-testid="stCaptionContainer"] p,
	[data-testid="stWidgetLabel"] p {
		color: #e4dccd !important;
	}
	h1, h2, h3 { color: #f4ecdd !important; font-family: 'Playfair Display', Georgia, serif; }
	h1 { font-size: 2.35rem; }
	h2 { font-size: 1.5rem; }
	[data-testid="stSidebar"] {
		background: linear-gradient(180deg, #151614 0%, #10110f 58%, #171512 100%);
		border-right: 1px solid rgba(218, 176, 93, .2);
	}
	[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
	[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #e8dfcf !important; }
	[data-testid="stSidebar"] [role="radiogroup"] label { padding: .45rem .55rem; border: 1px solid transparent; border-radius: 4px; }
	[data-testid="stSidebar"] [role="radiogroup"] label:hover { border-color: rgba(216, 173, 83, .22); background: rgba(216, 173, 83, .08); }
	[data-testid="stSidebar"] input[type="radio"] { accent-color: #e4b64f; }
	[data-testid="stMetric"] {
		padding: .9rem 1rem; border: 1px solid var(--wine-line);
		border-top: 2px solid var(--wine-copper); border-radius: 6px;
		background: var(--wine-panel);
	}
	[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #c9beab !important; }
	[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #f0d18b !important; font-family: 'Playfair Display', Georgia, serif; }
	.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] {
		border: 1px solid #e3b64d; border-radius: 4px; background: linear-gradient(110deg, #efc354, #c98616); color: #21190c; font-weight: 700;
	}
	.stButton > button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
		border: 1px solid #f0ce76; background: #f0c85c; color: #21190c;
	}
	[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] [role="combobox"] {
		border-color: rgba(234, 216, 183, .25); border-radius: 4px; background: #292a27; color: var(--wine-ink);
	}
	[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] input { color: #fff8e9 !important; }
	[data-baseweb="popover"], [data-baseweb="menu"] { border: 1px solid rgba(224, 187, 105, .25); background: #242522; }
	[data-baseweb="menu"] li, [role="option"] { background: #242522; color: #f2eadb; }
	[data-baseweb="menu"] li:hover, [role="option"]:hover { background: #3a3528; }
	[data-testid="stFileUploader"] section { border: 1px dashed rgba(226, 190, 112, .46); border-radius: 5px; background: rgba(29, 30, 27, .82); }
	[data-testid="stDataFrame"] { border: 1px solid var(--wine-line); border-radius: 5px; background: #1a1b19; }
	[data-testid="stDataFrame"] iframe { color-scheme: dark; }
	.page-kicker { margin-bottom: .35rem; color: #e5b64a; font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
	.page-description { max-width: 760px; margin: -.35rem 0 1.25rem; color: #d7cbb8 !important; font-size: .96rem; line-height: 1.55; }
	.page-rule { height: 1px; margin: 1.2rem 0 1.5rem; background: linear-gradient(90deg, rgba(224, 183, 83, .5), rgba(230, 205, 158, .16) 72%, transparent); }
	[data-testid="stMetric"] { box-shadow: 0 8px 25px rgba(0, 0, 0, .18); }
	[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] { filter: drop-shadow(0 8px 20px rgba(0, 0, 0, .14)); }
	.st-key-wine_hero {
		position: relative; overflow: hidden; padding: 2rem 2rem 1.55rem;
		border: 1px solid rgba(238, 191, 92, .32); border-radius: 10px;
		background-color: #171816;
		background-image: linear-gradient(90deg, rgba(10, 11, 10, .92) 0%, rgba(14, 15, 13, .78) 43%, rgba(18, 17, 14, .58) 100%),
			url('https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=2000&q=85');
		background-position: center 54%; background-size: cover;
		box-shadow: 0 18px 48px rgba(0, 0, 0, .34);
	}
	.st-key-wine_hero h1, .st-key-wine_hero h2, .st-key-wine_hero h3,
	.st-key-wine_hero p, .st-key-wine_hero label,
	.st-key-wine_hero [data-testid="stWidgetLabel"] p,
	.st-key-wine_hero [data-testid="stMarkdownContainer"] p { color: #fffaf0 !important; }
	.hero-eyebrow { margin-bottom: .7rem; color: #e8c57c !important; font-size: .7rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
	.hero-copy h1 { margin: .2rem 0 .75rem; color: #fffaf0 !important; font-size: clamp(2.4rem, 4.4vw, 4rem); line-height: 1.03; }
	.hero-copy h1 span { color: #e9b64c; }
	.hero-copy > p { max-width: 470px; color: rgba(255, 250, 240, .86) !important; font-size: .98rem; line-height: 1.65; }
	.hero-badge { display: inline-flex; align-items: center; gap: .45rem; margin-bottom: .7rem; padding: .38rem .65rem; border: 1px solid rgba(239, 214, 168, .32); border-radius: 20px; background: rgba(15, 17, 15, .4); color: #f3d48d; font-size: .72rem; }
	.hero-stats { display: flex; flex-wrap: wrap; gap: 2rem; margin-top: 1.45rem; }
	.hero-stat strong { display: block; color: #fffaf0; font-family: 'Playfair Display', Georgia, serif; font-size: 1.55rem; }
	.hero-stat span { color: rgba(255, 250, 240, .82); font-size: .72rem; }
	.st-key-wine_search_panel {
		padding: 1.2rem 1.25rem .9rem; border: 1px solid rgba(245, 210, 140, .28);
		border-radius: 8px; background: rgba(24, 25, 23, .83); backdrop-filter: blur(13px);
		box-shadow: 0 12px 34px rgba(0, 0, 0, .16);
	}
	.st-key-wine_search_panel h3 { margin-top: 0; font-size: 1.2rem; }
	.st-key-wine_search_panel [data-testid="stWidgetLabel"] p,
	.st-key-wine_search_panel label { color: rgba(255, 250, 240, .85) !important; }
	.st-key-wine_search_panel [data-testid="stSelectbox"] [role="combobox"] { border-color: rgba(246, 231, 203, .3); background: rgba(255, 255, 255, .1); color: #fffaf0; }
	.st-key-wine_search_panel [data-testid="stSelectbox"] input { color: #fffaf0 !important; }
	.st-key-wine_search_panel .stButton > button[kind="primary"] { width: 100%; min-height: 2.8rem; background: linear-gradient(110deg, #edbd48, #d98b17); color: #201a0f; font-weight: 700; }
	.st-key-wine_search_panel .stButton > button[kind="primary"]:hover { background: #f1c85f; color: #201a0f; }
	.hero-search-subtitle { margin: -.45rem 0 .65rem; color: rgba(255, 250, 240, .72) !important; font-size: .78rem; }
	.hero-popular { margin-top: .75rem; color: rgba(255, 250, 240, .78) !important; font-size: .72rem; }
	.step-caption { margin: -.35rem 0 .4rem; color: #e0c886; font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
	div[class*="st-key-step_"] button { min-height: 2.55rem; border-radius: 4px; font-size: .78rem; transition: background .18s ease, border-color .18s ease; }
	div[class*="st-key-step_"] button[kind="secondary"] { border: 1px solid rgba(226, 193, 126, .25); background: rgba(30, 31, 28, .9); color: #eee4d2; }
	div[class*="st-key-step_"] button[kind="secondary"]:hover { border-color: #c79b42; color: #fff; background: #282720; }
	div[class*="st-key-step_"] button[kind="primary"] { border: 1px solid #e0b64e; background: linear-gradient(110deg, #e6bb55, #b77b20); color: #20190d; }
	@media (max-width: 700px) {
		.block-container { padding-top: 1rem; }
		h1 { font-size: 1.9rem; }
		div[class*="st-key-step_"] button { min-height: 2.8rem; padding: .2rem; font-size: .65rem; }
		.st-key-wine_hero { padding: 1.15rem; background-position: 59% center; }
		.hero-copy h1 { font-size: 2.35rem; }
		.hero-stats { gap: 1.2rem; margin-top: 1rem; }
	}
	</style>
	""",
	unsafe_allow_html=True,
)


def navigate_to(page_name):
	st.session_state.active_page = page_name


def render_page_heading(section, title, description, current_page):
	st.markdown(f'<div class="page-kicker">{section}</div>', unsafe_allow_html=True)
	st.title(title)
	st.markdown(f'<p class="page-description">{description}</p>', unsafe_allow_html=True)
	st.markdown('<div class="page-rule"></div>', unsafe_allow_html=True)
	render_page_steps(current_page)


def render_page_steps(current_page):
	current_index = PAGES.index(current_page)
	st.markdown(
		f'<div class="step-caption">Workspace path / Step {current_index + 1:02d} of {len(PAGES):02d}</div>',
		unsafe_allow_html=True,
	)
	step_columns = st.columns(len(PAGES), gap="small")
	for index, (column, page_name) in enumerate(zip(step_columns, PAGES)):
		with column:
			column.button(
				f"{index + 1:02d}  {page_name}",
				type="primary" if page_name == current_page else "secondary",
				key=f"step_{page_name}",
				on_click=navigate_to,
				args=(page_name,),
				width="stretch",
			)
	st.markdown('<div class="page-rule"></div>', unsafe_allow_html=True)


def render_overview_hero(data):
	with st.container(key="wine_hero"):
		copy_column, search_column = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
		with copy_column:
			st.markdown('<div class="hero-badge">✦ &nbsp; Wine chemistry / K-Means</div>', unsafe_allow_html=True)
			st.markdown(
				'<div class="hero-copy"><h1>Discover your<br><span>wine profile.</span></h1><p>Explore the chemistry behind every sample and uncover the three natural groups in this classic wine collection.</p></div>',
				unsafe_allow_html=True,
			)
			st.markdown(
				f'<div class="hero-stats"><div class="hero-stat"><strong>{len(data):,}</strong><span>Wine samples</span></div><div class="hero-stat"><strong>{data["Wine_Cluster"].nunique()}</strong><span>Model groups</span></div><div class="hero-stat"><strong>{len(FEATURES)}</strong><span>Chemical features</span></div></div>',
				unsafe_allow_html=True,
			)
		with search_column:
			with st.container(key="wine_search_panel"):
				st.markdown("### Find a wine group")
				st.markdown('<p class="hero-search-subtitle">Explore the collection by cluster and chemistry.</p>', unsafe_allow_html=True)
				cluster_options = ["All groups"] + [f"Cluster {cluster}" for cluster in sorted(data["Wine_Cluster"].unique())]
				cluster_choice = st.selectbox("Model group", cluster_options, key="hero_cluster_choice")
				feature_options = ["Alcohol", "Color_Intensity", "Proline", "Flavanoids", "Hue", "OD280"]
				axis_columns = st.columns(2)
				x_choice = axis_columns[0].selectbox("Compare", feature_options, index=0, key="hero_x_feature")
				y_choice = axis_columns[1].selectbox("Against", feature_options, index=1, key="hero_y_feature")
				if st.button("Explore collection  →", type="primary", key="hero_explore"):
					st.session_state.overview_cluster_filter = cluster_choice
					st.session_state.overview_x_feature = x_choice
					st.session_state.overview_y_feature = y_choice
					st.session_state.overview_explore_requested = True
				st.markdown('<div class="hero-popular">Popular measurements / Alcohol · Color intensity · Proline</div>', unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
	model = joblib.load(BASE_DIR / "wine_model.joblib")
	scaler = joblib.load(BASE_DIR / "wine_scaler.joblib")
	return model, scaler


@st.cache_data
def load_reference_dataset():
	import kagglehub

	dataset_path = Path(kagglehub.dataset_download("harrywang/wine-dataset-for-clustering"))
	csv_files = list(dataset_path.rglob("wine-clustering.csv"))
	if not csv_files:
		raise FileNotFoundError("The Kaggle dataset did not contain wine-clustering.csv.")
	return pd.read_csv(csv_files[0])


def prepare_dataset(frame, model, scaler):
	missing = [feature for feature in FEATURES if feature not in frame.columns]
	if missing:
		raise ValueError("Missing columns: " + ", ".join(missing))
	measurements = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
	if measurements.isna().any().any():
		raise ValueError("The dataset contains missing or non-numeric feature values.")
	result = frame.copy()
	result["Wine_Cluster"] = model.predict(scaler.transform(measurements))
	return result


def get_dataset(model, scaler):
	if "wine_dataset" not in st.session_state:
		try:
			with st.spinner("Loading the wine dataset..."):
				st.session_state.wine_dataset = prepare_dataset(load_reference_dataset(), model, scaler)
			st.session_state.wine_source = "Kaggle wine clustering dataset"
		except Exception as error:
			st.error(f"Could not load the Kaggle dataset: {error}")
			return None
	return st.session_state.wine_dataset


def render_scatter_3d(data, model, scaler, x_feature, y_feature, z_feature):
	fig = px.scatter_3d(
		data,
		x=x_feature,
		y=y_feature,
		z=z_feature,
		color=data["Wine_Cluster"].astype(str),
		color_discrete_sequence=CLUSTER_COLORS,
		hover_data=["Alcohol", "Color_Intensity", "Proline", "Flavanoids"],
		labels={
			x_feature: FEATURE_LABELS[x_feature],
			y_feature: FEATURE_LABELS[y_feature],
			z_feature: FEATURE_LABELS[z_feature],
			"color": "Cluster",
		},
	)
	centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
	fig.add_trace(
		go.Scatter3d(
			x=centers[x_feature],
			y=centers[y_feature],
			z=centers[z_feature],
			mode="markers+text",
			name="Cluster centers",
			text=[f"Center {index}" for index in range(len(centers))],
			textposition="top center",
			marker=dict(size=8, symbol="diamond", color="#edc45e", line=dict(width=1, color="#fff0c8")),
			textfont=dict(color="#f7e6bd"),
			hovertemplate="%{text}<extra></extra>",
		)
	)
	fig.update_layout(
		height=560,
		margin=dict(l=0, r=0, t=12, b=0),
		legend_title_text="Cluster",
		paper_bgcolor="rgba(0,0,0,0)",
		font=dict(family="DM Sans, sans-serif", color="#eee5d5"),
		legend=dict(font=dict(color="#eee5d5"), bgcolor="rgba(20,21,19,.55)"),
		scene=dict(
			bgcolor="rgba(20,21,19,.38)",
			xaxis_title=FEATURE_LABELS[x_feature],
			yaxis_title=FEATURE_LABELS[y_feature],
			zaxis_title=FEATURE_LABELS[z_feature],
			xaxis=dict(color="#ded3bf", gridcolor="#49453b", backgroundcolor="rgba(20,21,19,.18)"),
			yaxis=dict(color="#ded3bf", gridcolor="#49453b", backgroundcolor="rgba(20,21,19,.18)"),
			zaxis=dict(color="#ded3bf", gridcolor="#49453b", backgroundcolor="rgba(20,21,19,.18)"),
		),
	)
	st.plotly_chart(fig, width="stretch")


def overview_page(model, scaler):
	data = get_dataset(model, scaler)
	if data is None:
		return

	render_overview_hero(data)
	render_page_steps("Overview")

	chart_data = data
	cluster_filter = st.session_state.get("overview_cluster_filter", "All groups")
	if cluster_filter != "All groups":
		cluster_id = int(cluster_filter.removeprefix("Cluster "))
		chart_data = data[data["Wine_Cluster"] == cluster_id]
	x_feature = st.session_state.get("overview_x_feature", "Alcohol")
	y_feature = st.session_state.get("overview_y_feature", "Color_Intensity")
	st.subheader("Explore the collection")
	st.caption(
		f"Showing {len(chart_data):,} of {len(data):,} samples / "
		f"{cluster_filter} / {FEATURE_LABELS[x_feature]} vs {FEATURE_LABELS[y_feature]}"
	)

	chart_column, summary_column = st.columns([1.5, 1])
	with chart_column:
		fig = px.scatter(
			chart_data,
			x=x_feature,
			y=y_feature,
			color=chart_data["Wine_Cluster"].astype(str),
			color_discrete_sequence=CLUSTER_COLORS,
			labels={x_feature: FEATURE_LABELS[x_feature], y_feature: FEATURE_LABELS[y_feature], "color": "Cluster"},
			hover_data=["Flavanoids", "Hue", "Proline"],
		)
		fig.update_layout(
			height=420,
			margin=dict(l=8, r=8, t=16, b=8),
			paper_bgcolor="rgba(0,0,0,0)",
			plot_bgcolor="rgba(0,0,0,0)",
			font=dict(family="DM Sans, sans-serif", color="#eee5d5"),
			legend_title_text="Cluster",
			legend=dict(font=dict(color="#eee5d5"), bgcolor="rgba(20,21,19,.55)"),
		)
		fig.update_xaxes(showgrid=True, gridcolor="#393a35", zeroline=False, color="#d9cfbf")
		fig.update_yaxes(showgrid=True, gridcolor="#393a35", zeroline=False, color="#d9cfbf")
		st.plotly_chart(fig, width="stretch")
	with summary_column:
		st.subheader("Cluster sizes")
		counts = chart_data["Wine_Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Samples")
		bar = px.bar(
			counts,
			x="Cluster",
			y="Samples",
			color=counts["Cluster"].astype(str),
			color_discrete_sequence=CLUSTER_COLORS,
			text="Samples",
		)
		bar.update_layout(
			height=310,
			showlegend=False,
			margin=dict(l=5, r=5, t=12, b=5),
			paper_bgcolor="rgba(0,0,0,0)",
			plot_bgcolor="rgba(0,0,0,0)",
			font=dict(family="DM Sans, sans-serif", color="#e8dfcf"),
		)
		bar.update_xaxes(showgrid=False, color="#d6cbb7", title="Cluster")
		bar.update_yaxes(showgrid=True, gridcolor="#383a35", color="#d6cbb7", title="Samples")
		bar.update_traces(textfont_color="#fff5df")
		st.plotly_chart(bar, width="stretch")
		st.dataframe(counts, hide_index=True, width="stretch")
	st.caption("Cluster numbers describe chemical similarity; they are not wine quality ratings. Data source: " + st.session_state.get("wine_source", "Wine dataset"))


def prediction_page(model, scaler):
	render_page_heading(
		"Model / Inference",
		"Predict a wine cluster",
		"Enter the 13 chemical measurements to locate the sample against the trained cluster centers.",
		"Predict a wine",
	)
	with st.form("wine_prediction"):
		values = {}
		for start in range(0, len(FEATURES), 3):
			columns = st.columns(3)
			for column, feature in zip(columns, FEATURES[start : start + 3]):
				with column:
					values[feature] = st.number_input(
						FEATURE_LABELS[feature],
						value=float(DEFAULT_SAMPLE[feature]),
						format="%.2f",
					)
		submitted = st.form_submit_button("Predict cluster", type="primary")

	if submitted:
		sample = pd.DataFrame([values], columns=FEATURES)
		scaled = scaler.transform(sample)
		cluster = int(model.predict(scaled)[0])
		distance = float(model.transform(scaled)[0][cluster])
		st.success(f"This wine belongs to Cluster {cluster}.")
		result_columns = st.columns(2)
		result_columns[0].metric("Predicted cluster", f"Cluster {cluster}")
		result_columns[1].metric("Distance to cluster center", f"{distance:.3f}")
		st.caption("The cluster is a model-generated group, not a quality score or grape variety.")


def cluster_map_page(model, scaler):
	data = get_dataset(model, scaler)
	render_page_heading(
		"Explore / Three dimensions",
		"3D cluster map",
		"Choose three chemical features to inspect the wine samples and K-Means centers in an interactive 3D view.",
		"3D cluster map",
	)
	if data is None:
		return

	columns = st.columns(3)
	x_feature = columns[0].selectbox("X axis", FEATURES, index=0)
	y_feature = columns[1].selectbox("Y axis", FEATURES, index=9)
	z_feature = columns[2].selectbox("Z axis", FEATURES, index=12)
	render_scatter_3d(data, model, scaler, x_feature, y_feature, z_feature)


def cluster_profiles_page(model, scaler):
	data = get_dataset(model, scaler)
	render_page_heading(
		"Compare / Chemical signatures",
		"Cluster profiles",
		"Compare each cluster's average measurements with the overall dataset to see which features distinguish the groups.",
		"Cluster profiles",
	)
	if data is None:
		return

	means = data.groupby("Wine_Cluster")[FEATURES].mean().sort_index()
	standardized = (means - means.mean()) / means.std().fillna(1).replace(0, 1)
	labels = [FEATURE_LABELS[feature] for feature in FEATURES]
	fig = px.imshow(
		standardized,
		x=labels,
		y=[f"Cluster {cluster}" for cluster in standardized.index],
		color_continuous_scale="RdBu",
		color_continuous_midpoint=0,
		aspect="auto",
		labels={"x": "Chemical feature", "y": "Cluster", "color": "Relative level"},
	)
	fig.update_layout(
		height=420,
		margin=dict(l=5, r=5, t=15, b=100),
		font=dict(family="DM Sans, sans-serif", color="#eee5d5"),
		paper_bgcolor="rgba(0,0,0,0)",
	)
	fig.update_xaxes(tickangle=-35, tickfont=dict(color="#d9cfbf"), title_font=dict(color="#eee5d5"))
	fig.update_yaxes(tickfont=dict(color="#eee5d5"), title_font=dict(color="#eee5d5"))
	st.plotly_chart(fig, width="stretch")
	st.subheader("Average measurements")
	st.dataframe(means.rename(index=lambda cluster: f"Cluster {cluster}"), width="stretch")


def data_studio_page(model, scaler):
	data = get_dataset(model, scaler)
	render_page_heading(
		"Data / Import and export",
		"Data studio",
		"Use the public Kaggle collection or upload a CSV containing all 13 chemical measurement columns.",
		"Data studio",
	)
	uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
	button_column, source_column = st.columns([1, 2])
	apply_upload = button_column.button("Use uploaded dataset", disabled=uploaded_file is None)
	if apply_upload and uploaded_file is not None:
		try:
			data = prepare_dataset(pd.read_csv(uploaded_file), model, scaler)
			st.session_state.wine_dataset = data
			st.session_state.wine_source = uploaded_file.name
			st.success(f"Loaded {len(data):,} wine samples.")
		except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
			st.error(f"Could not use this CSV: {error}")
			data = None
	if source_column.button("Restore Kaggle dataset"):
		st.session_state.pop("wine_dataset", None)
		st.session_state.wine_source = "Kaggle wine clustering dataset"
		st.rerun()

	data = st.session_state.get("wine_dataset", data)
	if data is None:
		return

	counts = data["Wine_Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Samples")
	st.caption(f"Source: {st.session_state.get('wine_source', 'Wine dataset')} / {len(data):,} samples")
	st.dataframe(counts, hide_index=True, width="stretch")
	st.dataframe(data.head(20), hide_index=True, width="stretch")
	st.download_button(
		"Download clustered CSV",
		data=data.to_csv(index=False).encode("utf-8"),
		file_name="wine_clusters.csv",
		mime="text/csv",
	)


model, scaler = load_artifacts()

with st.sidebar:
	st.markdown("### Wine Cluster Studio")
	st.caption("K-Means / Wine chemistry")
	page = st.radio(
		"Pages",
		PAGES,
		label_visibility="collapsed",
		key="active_page",
	)
	st.divider()
	active_data = st.session_state.get("wine_dataset")
	if active_data is not None:
		st.metric("Active samples", f"{len(active_data):,}")
		st.caption(st.session_state.get("wine_source", "Active dataset"))

if page == "Overview":
	overview_page(model, scaler)
elif page == "Predict a wine":
	prediction_page(model, scaler)
elif page == "3D cluster map":
	cluster_map_page(model, scaler)
elif page == "Cluster profiles":
	cluster_profiles_page(model, scaler)
else:
	data_studio_page(model, scaler)
