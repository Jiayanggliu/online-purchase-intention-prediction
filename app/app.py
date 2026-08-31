from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "online_shoppers_intention.csv"

# -------------------------
# Data + portfolio model
# -------------------------
df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

categorical_columns = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]
numeric_cols = X.select_dtypes(include="number").columns
continuous_columns = [c for c in numeric_cols if c not in categorical_columns]

for col in continuous_columns:
    X[col] = np.log1p(X[col])

X = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=int)

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop = set()
for col in upper.columns:
    for row in upper.index:
        value = upper.loc[row, col]
        if pd.notna(value) and value > 0.80:
            cols_to_drop.add(row)
X = X.drop(columns=list(cols_to_drop))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
PREDICTION_THRESHOLD = 0.62

feature_importance = (
    pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .head(10)
    .sort_values("Importance")
)
fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Features Predicting Purchase Intention",
)
fig_importance.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20))

month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def predict_purchase(admin_duration, info_duration, product_duration, exit_rate,
                     page_value, special_day, month, visitor_type, weekend):
    input_row = pd.DataFrame(np.zeros((1, len(X_train.columns))), columns=X_train.columns)

    numeric_inputs = {
        "Administrative_Duration": admin_duration or 0,
        "Informational_Duration": info_duration or 0,
        "ProductRelated_Duration": product_duration or 0,
        "ExitRates": exit_rate or 0,
        "PageValues": page_value or 0,
        "SpecialDay": special_day or 0,
    }
    for col, value in numeric_inputs.items():
        if col in input_row.columns:
            input_row[col] = np.log1p(max(float(value), 0))

    for col in [f"Month_{month}", f"VisitorType_{visitor_type}"]:
        if col in input_row.columns:
            input_row[col] = 1
    if weekend and "Weekend_True" in input_row.columns:
        input_row["Weekend_True"] = 1

    probability = model.predict_proba(input_row)[0, 1]
    return probability, int(probability > PREDICTION_THRESHOLD)


app = Dash(__name__)
server = app.server

CARD = {
    "background": "#ffffff",
    "border": "1px solid #e8eee9",
    "borderRadius": "18px",
    "padding": "20px",
    "boxShadow": "0 10px 30px rgba(31, 48, 41, 0.05)",
}

app.layout = html.Div([
    html.Div([
        html.Div("PURCHASE INTENTION · ML DASHBOARD", style={"fontSize": "12px", "letterSpacing": "2px", "fontWeight": 700}),
        html.H1("What makes a browsing session convert?", style={"fontSize": "42px", "margin": "10px 0 8px"}),
        html.P("Explore conversion behavior and estimate purchase probability with the portfolio Random Forest model.",
               style={"maxWidth": "760px", "fontSize": "17px", "color": "#5c685f"}),
    ], style={"padding": "36px", "background": "#dff7ea", "borderRadius": "28px", "marginBottom": "24px"}),

    html.Div([
        html.Div([html.Div("12,205", style={"fontSize": "30px", "fontWeight": 700}), html.Div("Sessions")], style=CARD),
        html.Div([html.Div("1,908", style={"fontSize": "30px", "fontWeight": 700}), html.Div("Purchases")], style=CARD),
        html.Div([html.Div("15.63%", style={"fontSize": "30px", "fontWeight": 700}), html.Div("Conversion rate")], style=CARD),
        html.Div([html.Div("0.688", style={"fontSize": "30px", "fontWeight": 700}), html.Div("Random Forest F1")], style=CARD),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "14px", "marginBottom": "24px"}),

    html.Div([
        html.Div([
            html.H3("Explore conversion"),
            html.Label("Month"),
            dcc.Dropdown(["All"] + month_order, "All", id="month-filter", clearable=False),
            html.Br(),
            html.Label("Visitor type"),
            dcc.Dropdown(["All", "New_Visitor", "Returning_Visitor", "Other"], "All", id="visitor-filter", clearable=False),
        ], style=CARD),
        html.Div(dcc.Graph(id="month-graph", config={"displayModeBar": False}), style=CARD),
        html.Div(dcc.Graph(id="visitor-graph", config={"displayModeBar": False}), style=CARD),
    ], style={"display": "grid", "gridTemplateColumns": "minmax(220px,.7fr) 1.15fr 1.15fr", "gap": "14px", "marginBottom": "24px"}),

    html.Div(dcc.Graph(figure=fig_importance, config={"displayModeBar": False}), style={**CARD, "marginBottom": "24px"}),

    html.Div([
        html.H2("Purchase predictor", style={"marginTop": 0}),
        html.P("Try a hypothetical session. This is a portfolio demo, not a production decision system.", style={"color": "#68746b"}),
        html.Div([
            dcc.Input(id="admin-duration", type="number", value=100, placeholder="Administrative duration"),
            dcc.Input(id="info-duration", type="number", value=50, placeholder="Informational duration"),
            dcc.Input(id="product-duration", type="number", value=500, placeholder="Product duration"),
            dcc.Input(id="exit-rate", type="number", value=0.05, step=0.01, placeholder="Exit rate"),
            dcc.Input(id="page-value", type="number", value=40, placeholder="Page value"),
            dcc.Input(id="special-day", type="number", value=0, step=0.1, placeholder="Special day"),
            dcc.Dropdown(month_order, "Nov", id="prediction-month", clearable=False),
            dcc.Dropdown(["New_Visitor", "Returning_Visitor", "Other"], "Returning_Visitor", id="prediction-visitor", clearable=False),
            dcc.Dropdown([{"label":"Weekend", "value":True}, {"label":"Weekday", "value":False}], True, id="prediction-weekend", clearable=False),
        ], style={"display":"grid", "gridTemplateColumns":"repeat(auto-fit, minmax(210px, 1fr))", "gap":"12px"}),
        html.Button("Predict purchase", id="predict-button", n_clicks=0,
                    style={"marginTop":"18px", "border":0, "borderRadius":"999px", "padding":"12px 20px", "background":"#1f3d31", "color":"white", "fontWeight":700}),
        html.Div("Enter session information and click Predict purchase.", id="prediction-output",
                 style={"marginTop":"18px", "fontSize":"20px", "fontWeight":700}),
    ], style={**CARD, "padding":"28px"}),
], style={"maxWidth": "1180px", "margin": "0 auto", "padding": "28px 18px 60px", "fontFamily": "Inter, system-ui, sans-serif", "background": "#fbfcfa", "color": "#1e2c25"})


@app.callback(
    Output("month-graph", "figure"), Output("visitor-graph", "figure"),
    Input("month-filter", "value"), Input("visitor-filter", "value")
)
def update_charts(selected_month, selected_visitor):
    filtered = df.copy()
    if selected_month != "All":
        filtered = filtered[filtered["Month"] == selected_month]
    if selected_visitor != "All":
        filtered = filtered[filtered["VisitorType"] == selected_visitor]

    monthly = filtered.groupby("Month", observed=False)["Revenue"].mean().mul(100).reset_index(name="Conversion Rate")
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")
    month_fig = px.bar(monthly, x="Month", y="Conversion Rate", title="Conversion rate by month")

    visitor = filtered.groupby("VisitorType", observed=False)["Revenue"].mean().mul(100).reset_index(name="Conversion Rate")
    visitor_fig = px.bar(visitor, x="VisitorType", y="Conversion Rate", title="Conversion rate by visitor type")

    for fig in [month_fig, visitor_fig]:
        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20))
    return month_fig, visitor_fig


@app.callback(
    Output("prediction-output", "children"), Input("predict-button", "n_clicks"),
    State("admin-duration", "value"), State("info-duration", "value"),
    State("product-duration", "value"), State("exit-rate", "value"),
    State("page-value", "value"), State("special-day", "value"),
    State("prediction-month", "value"), State("prediction-visitor", "value"),
    State("prediction-weekend", "value"),
)
def update_prediction(n_clicks, admin_duration, info_duration, product_duration, exit_rate,
                      page_value, special_day, month, visitor_type, weekend):
    if not n_clicks:
        return "Enter session information and click Predict purchase."
    probability, prediction = predict_purchase(admin_duration, info_duration, product_duration,
                                               exit_rate, page_value, special_day, month,
                                               visitor_type, weekend)
    label = "Likely purchase" if prediction else "Unlikely purchase"
    return f"{label} · estimated probability {probability * 100:.1f}%"


if __name__ == "__main__":
    app.run(debug=True)
