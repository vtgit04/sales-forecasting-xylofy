# ─── app.py — Sales Forecasting Dashboard ───────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Sales Intelligence Dashboard',
    page_icon='📊',
    layout='wide'
)

# ── Load & Prepare Data (cached so it only runs once) ────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['Quarter']    = df['Order Date'].dt.quarter
    df['ShipDays']   = (df['Ship Date'] - df['Order Date']).dt.days

    monthly = df.groupby(
        df['Order Date'].dt.to_period('M')
    )['Sales'].sum()
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index()

    weekly = df.groupby(
        df['Order Date'].dt.to_period('W')
    )['Sales'].sum()
    weekly.index = weekly.index.to_timestamp()
    weekly = weekly.sort_index()

    return df, monthly, weekly

df, monthly, weekly = load_data()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title('📊 Sales Intelligence')
st.sidebar.markdown('**XYLOFY AI Internship — Week 3&4**')
st.sidebar.markdown('*Vijay Tiwari*')
st.sidebar.markdown('---')

page = st.sidebar.radio('Navigate to:', [
    '📈 Sales Overview',
    '🔮 Forecast Explorer',
    '🚨 Anomaly Report',
    '🎯 Demand Segments'
])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == '📈 Sales Overview':
    st.title('📈 Sales Overview Dashboard')
    st.markdown('4 years of Superstore sales data — 2015 to 2018')

    # ── KPI Cards ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Revenue',    f"${df['Sales'].sum():,.0f}")
    col2.metric('Total Orders',     f"{df['Order ID'].nunique():,}")
    col3.metric('Avg Order Value',  f"${df['Sales'].mean():,.0f}")
    col4.metric('Avg Ship Time',    f"{df['ShipDays'].mean():.1f} days")

    st.markdown('---')

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    selected_region = col_f1.multiselect(
        'Filter by Region:',
        options=df['Region'].unique().tolist(),
        default=df['Region'].unique().tolist()
    )
    selected_category = col_f2.multiselect(
        'Filter by Category:',
        options=df['Category'].unique().tolist(),
        default=df['Category'].unique().tolist()
    )

    # Apply filters
    df_filtered = df[
        df['Region'].isin(selected_region) &
        df['Category'].isin(selected_category)
    ]

    st.markdown('---')

    # ── Chart 1: Sales by Year ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Total Sales by Year')
        yearly = df_filtered.groupby('Year')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#3498db','#2ecc71','#e67e22','#e74c3c']
        bars = ax.bar(yearly.index.astype(str), yearly.values/1000,
                      color=colors, edgecolor='white')
        for bar, val in zip(bars, yearly.values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1,
                    f'${val/1000:.0f}K',
                    ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('Sales ($K)')
        ax.set_title('Annual Revenue')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader('Sales by Category')
        cat_sales = df_filtered.groupby('Category')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors2 = ['#e74c3c','#3498db','#2ecc71']
        ax.pie(cat_sales.values, labels=cat_sales.index,
               autopct='%1.1f%%', colors=colors2,
               startangle=90, wedgeprops={'edgecolor':'white'})
        ax.set_title('Revenue Share by Category')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Chart 2: Monthly Trend ────────────────────────────────────────────────
    st.subheader('Monthly Sales Trend')
    monthly_filtered = df_filtered.groupby(
        df_filtered['Order Date'].dt.to_period('M')
    )['Sales'].sum()
    monthly_filtered.index = monthly_filtered.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly_filtered.index, monthly_filtered.values/1000,
            color='steelblue', linewidth=2, marker='o', markersize=3)
    ax.fill_between(monthly_filtered.index,
                    monthly_filtered.values/1000,
                    alpha=0.15, color='steelblue')
    ax.set_ylabel('Sales ($K)')
    ax.set_xlabel('Month')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Chart 3: Sales by Region ──────────────────────────────────────────────
    st.subheader('Sales by Region')
    region_yr = df_filtered.groupby(
        ['Region','Year']
    )['Sales'].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    region_yr.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white')
    ax.set_ylabel('Sales ($)')
    ax.set_xlabel('Region')
    ax.legend(title='Year', bbox_to_anchor=(1,1))
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == '🔮 Forecast Explorer':
    st.title('🔮 Forecast Explorer')
    st.markdown('XGBoost forecasting — select a segment and horizon')

    # ── Helper functions ──────────────────────────────────────────────────────
    def make_xgb_features(series):
        df_feat = pd.DataFrame({'y': series})
        df_feat['lag1'] = df_feat['y'].shift(1)
        df_feat['lag2'] = df_feat['y'].shift(2)
        df_feat['lag3'] = df_feat['y'].shift(3)
        df_feat['rolling_mean3'] = df_feat['y'].shift(1).rolling(3).mean()
        df_feat['month']   = series.index.month
        df_feat['quarter'] = series.index.quarter
        df_feat['season']  = df_feat['month'].map({
            12:4,1:4,2:4, 3:1,4:1,5:1,
            6:2,7:2,8:2,  9:3,10:3,11:3
        })
        return df_feat.dropna()

    def run_xgb_forecast(series, horizon):
        feat = make_xgb_features(series)
        feat_cols = ['lag1','lag2','lag3',
                     'rolling_mean3','month','quarter','season']
        X = feat[feat_cols]
        y = feat['y']
        model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, random_state=42, verbosity=0
        )
        model.fit(X, y)

        # Iteratively forecast future months
        history = series.copy()
        preds   = []
        for i in range(horizon):
            tmp  = make_xgb_features(history)
            Xnew = tmp[feat_cols].iloc[[-1]]
            pred = model.predict(Xnew)[0]
            preds.append(pred)
            # Append prediction to history for next iteration
            next_date = history.index[-1] + pd.DateOffset(months=1)
            history   = pd.concat([
                history,
                pd.Series([pred], index=[next_date])
            ])
        return preds

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    segment_type = col1.selectbox(
        'Select Segment Type:',
        ['Overall', 'Category', 'Region']
    )

    if segment_type == 'Overall':
        series = monthly.copy()
        seg_label = 'Overall'
    elif segment_type == 'Category':
        cat = col2.selectbox('Select Category:',
                             df['Category'].unique().tolist())
        seg = df[df['Category'] == cat]
        series = seg.groupby(
            seg['Order Date'].dt.to_period('M')
        )['Sales'].sum()
        series.index = series.index.to_timestamp()
        series = series.sort_index()
        seg_label = cat
    else:
        reg = col2.selectbox('Select Region:',
                             df['Region'].unique().tolist())
        seg = df[df['Region'] == reg]
        series = seg.groupby(
            seg['Order Date'].dt.to_period('M')
        )['Sales'].sum()
        series.index = series.index.to_timestamp()
        series = series.sort_index()
        seg_label = reg

    horizon = st.slider(
        'Forecast Horizon (months ahead):',
        min_value=1, max_value=3, value=3
    )

    # ── Run forecast ──────────────────────────────────────────────────────────
    if st.button('🔮 Generate Forecast'):
        with st.spinner('Running XGBoost forecast...'):
            preds = run_xgb_forecast(series, horizon)

            # Future dates
            future_dates = [
                series.index[-1] + pd.DateOffset(months=i+1)
                for i in range(horizon)
            ]

            # Plot
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(series.index, series.values/1000,
                    color='steelblue', linewidth=2,
                    label='Historical Sales')
            ax.plot(future_dates, np.array(preds)/1000,
                    color='crimson', linewidth=2.5,
                    marker='o', markersize=9,
                    linestyle='--', label='Forecast')

            for date, val in zip(future_dates, preds):
                ax.annotate(f'${val/1000:.1f}K',
                            xy=(date, val/1000),
                            xytext=(0, 12),
                            textcoords='offset points',
                            ha='center', fontsize=9,
                            color='crimson', fontweight='bold')

            ax.axvspan(series.index[-1], future_dates[-1],
                       alpha=0.08, color='crimson')
            ax.set_title(f'XGBoost Forecast — {seg_label}',
                         fontsize=13, fontweight='bold')
            ax.set_ylabel('Sales ($K)')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Forecast table
            st.subheader('📋 Forecast Values')
            forecast_df = pd.DataFrame({
                'Month'            : [d.strftime('%b %Y')
                                      for d in future_dates],
                'Forecasted Sales' : [f'${p:,.0f}' for p in preds]
            })
            st.dataframe(forecast_df, use_container_width=True)

            # Model metrics
            st.subheader('📊 Model Performance (on last 3 months of training data)')
            train_s = series[:-3]
            test_s  = series[-3:]
            feat    = make_xgb_features(series)
            feat_cols = ['lag1','lag2','lag3',
                         'rolling_mean3','month','quarter','season']
            model2  = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=3, random_state=42, verbosity=0
            )
            model2.fit(feat.iloc[:-3][feat_cols], feat.iloc[:-3]['y'])
            test_preds = model2.predict(feat.iloc[-3:][feat_cols])
            mae  = np.mean(np.abs(test_s.values - test_preds))
            rmse = np.sqrt(np.mean((test_s.values - test_preds)**2))
            mc1, mc2 = st.columns(2)
            mc1.metric('MAE',  f'${mae:,.0f}')
            mc2.metric('RMSE', f'${rmse:,.0f}')

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == '🚨 Anomaly Report':
    st.title('🚨 Anomaly Report')
    st.markdown('Unusual sales weeks detected via Isolation Forest + Z-Score')

    # Rebuild anomaly detection
    weekly_df = pd.DataFrame({
        'Date' : weekly.index,
        'Sales': weekly.values
    })

    # Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    weekly_df['IF_Anomaly'] = (
        iso.fit_predict(weekly.values.reshape(-1,1)) == -1
    )

    # Z-Score
    g_mean = weekly.mean()
    g_std  = weekly.std()
    weekly_df['ZScore']     = ((weekly - g_mean) / g_std).values
    weekly_df['ZS_Anomaly'] = (weekly_df['ZScore'].abs() > 2)

    # Both agree
    both_mask = weekly_df['IF_Anomaly'] & weekly_df['ZS_Anomaly']
    both_df   = weekly_df[both_mask]

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric('Weeks Analyzed',          f'{len(weekly_df)}')
    c2.metric('IF Anomalies',            f'{weekly_df["IF_Anomaly"].sum()}')
    c3.metric('Confirmed (Both Methods)',f'{both_mask.sum()}')

    st.markdown('---')

    # Anomaly chart
    st.subheader('📈 Weekly Sales with Anomalies Highlighted')
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weekly_df['Date'], weekly_df['Sales']/1000,
            color='steelblue', linewidth=1.2, alpha=0.7,
            label='Weekly Sales')
    ax.axhspan((g_mean - 2*g_std)/1000,
               (g_mean + 2*g_std)/1000,
               alpha=0.08, color='green',
               label='Normal Range (±2 std)')
    ax.axhline(g_mean/1000, color='gray',
               linestyle='--', linewidth=1,
               label=f'Mean (${g_mean/1000:.1f}K)')

    if_only = weekly_df[weekly_df['IF_Anomaly'] & ~weekly_df['ZS_Anomaly']]
    ax.scatter(if_only['Date'], if_only['Sales']/1000,
               color='darkorange', s=70, zorder=5,
               marker='v', label='IF Only')
    ax.scatter(both_df['Date'], both_df['Sales']/1000,
               color='crimson', s=120, zorder=6,
               marker='^', label='Both Methods ⚠️')

    ax.set_ylabel('Sales ($K)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('---')

    # Anomaly table
    st.subheader('📋 Confirmed Anomaly Dates (Both Methods)')
    anomaly_table = both_df[['Date','Sales','ZScore']].copy()
    anomaly_table['Date']      = anomaly_table['Date'].dt.strftime('%b %d, %Y')
    anomaly_table['Sales']     = anomaly_table['Sales'].apply(
                                     lambda x: f'${x:,.0f}')
    anomaly_table['ZScore']    = anomaly_table['ZScore'].apply(
                                     lambda x: f'{x:+.2f}')
    anomaly_table['Direction'] = both_df['Sales'].apply(
        lambda x: '📈 Spike' if x > g_mean else '📉 Drop'
    ).values
    anomaly_table.columns = ['Week', 'Sales', 'Z-Score', 'Direction']
    st.dataframe(anomaly_table, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMAND SEGMENTS
# ════════════════════════════════════════════════════════════════════════════
elif page == '🎯 Demand Segments':
    st.title('🎯 Product Demand Segments')
    st.markdown('K-Means clustering of sub-categories by demand behavior')

    # Rebuild features
    total_sales  = df.groupby('Sub-Category')['Sales'].sum()
    sales_by_yr  = df.groupby(
        ['Sub-Category','Year']
    )['Sales'].sum().unstack()
    yoy_growth   = ((sales_by_yr[2018] - sales_by_yr[2017]) /
                     sales_by_yr[2017] * 100).round(2)
    monthly_sub  = df.groupby([
        df['Order Date'].dt.to_period('M'), 'Sub-Category'
    ])['Sales'].sum().unstack().fillna(0)
    volatility   = monthly_sub.std()
    avg_order    = df.groupby('Sub-Category')['Sales'].mean()

    features = pd.DataFrame({
        'TotalSales' : total_sales,
        'YoYGrowth'  : yoy_growth,
        'Volatility' : volatility,
        'AvgOrderVal': avg_order
    }).dropna()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['Cluster'] = km.fit_predict(X_scaled)

    cluster_labels = {
        0: 'High Growth, Low Volume',
        1: 'Stagnant / Declining',
        2: 'High Value, Volatile',
        3: 'High Volume, Growing'
    }
    features['ClusterLabel'] = features['Cluster'].map(cluster_labels)

    # PCA plot
    st.subheader('🗺️ Cluster Map (PCA View)')
    pca    = PCA(n_components=2)
    X_pca  = pca.fit_transform(X_scaled)
    colors = {0:'#2ecc71', 1:'#3498db', 2:'#e74c3c', 3:'#e67e22'}

    fig, ax = plt.subplots(figsize=(11, 7))
    for cluster, label in cluster_labels.items():
        mask = features['Cluster'] == cluster
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[cluster], s=200,
                   label=f'{label}',
                   edgecolors='white', linewidth=0.8, zorder=5)
        for idx, (i, _) in enumerate(features[mask].iterrows()):
            ax.annotate(i,
                        xy=(X_pca[mask][idx,0], X_pca[mask][idx,1]),
                        xytext=(6, 4), textcoords='offset points',
                        fontsize=8.5, fontweight='bold',
                        color=colors[cluster])
    ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend(fontsize=9)
    ax.set_title('Product Demand Segmentation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('---')

    # Cluster table
    st.subheader('📋 Sub-Category Cluster Assignments')
    colors_hex = {
        'High Growth, Low Volume' : '🟢',
        'Stagnant / Declining'    : '🔵',
        'High Value, Volatile'    : '🔴',
        'High Volume, Growing'    : '🟠'
    }
    display_df = features[['ClusterLabel','TotalSales',
                            'YoYGrowth','Volatility',
                            'AvgOrderVal']].copy()
    display_df['ClusterLabel'] = display_df['ClusterLabel'].apply(
        lambda x: f'{colors_hex[x]} {x}'
    )
    display_df.columns = ['Cluster','Total Sales',
                          'YoY Growth %','Volatility','Avg Order $']
    display_df['Total Sales']  = display_df['Total Sales'].apply(
                                     lambda x: f'${x:,.0f}')
    display_df['Avg Order $']  = display_df['Avg Order $'].apply(
                                     lambda x: f'${x:,.0f}')
    display_df['YoY Growth %'] = display_df['YoY Growth %'].apply(
                                     lambda x: f'{x:+.1f}%')
    display_df['Volatility']   = display_df['Volatility'].apply(
                                     lambda x: f'{x:,.0f}')
    st.dataframe(display_df, use_container_width=True)

    # Stocking strategy
    st.markdown('---')
    st.subheader('💡 Recommended Stocking Strategy')
    strategies = {
        '🟢 High Growth, Low Volume' :
            'Gradually increase stock each quarter. Low risk — demand is stable and rising.',
        '🔵 Stagnant / Declining'    :
            'Maintain lean inventory. Use promotions to clear slow movers. Review for discontinuation.',
        '🔴 High Value, Volatile'    :
            'Avoid large standing stock. Use just-in-time ordering. Maintain strong supplier relationships.',
        '🟠 High Volume, Growing'    :
            'Highest priority. Maintain 6–8 weeks safety stock. Critical to avoid stockouts in Q4.'
    }
    for cluster, strategy in strategies.items():
        st.info(f'**{cluster}** — {strategy}')