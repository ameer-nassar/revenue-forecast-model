# ================================
# Revenue Forecasting Model - Executive Summary & Visuals
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker

# ================================
# Load & Prepare Data
# ================================

df = pd.read_csv("revenue_forecasting_dataset.csv")

# Convert Month to datetime and numeric for sorting
df['Month_dt'] = pd.to_datetime(df['Month'])  # Let pandas infer format
df['Month_Num'] = df['Month_dt'].dt.month
df.sort_values(by=['Department', 'Month_dt'], inplace=True)

# ================================
# Executive Summary Metrics
# ================================

total_budget = df['Budget'].sum()
total_forecast = df['Forecast'].sum()
total_actual = df['Actual'].sum()
total_variance_amt = total_actual - total_forecast
total_variance_pct = (total_variance_amt / total_forecast) * 100

print("====== Revenue Forecasting Executive Summary ======")
print(f"Total Budget:   ${total_budget:,.2f}")
print(f"Total Forecast: ${total_forecast:,.2f}")
print(f"Total Actual:   ${total_actual:,.2f}")
print(f"Total Variance: {total_variance_amt:,.2f} USD ({total_variance_pct:+.2f}%)\n")

# Department-level summary
print("===== Department-Level Forecast Summary =====")
grouped = df.groupby('Department')[['Forecast', 'Actual']].sum()
grouped['pct_variance'] = ((grouped['Actual'] - grouped['Forecast']) / grouped['Forecast']) * 100

for dept, row in grouped.iterrows():
    status = "over" if row['pct_variance'] > 0 else "under"
    print(f"{dept:16} is {abs(row['pct_variance']):.1f}% {status} forecast "
          f"({row['Actual'] - row['Forecast']:,.0f} USD)")

# ================================
# Plot 1: Actual vs Forecast by Department (Monthly)
# ================================

sns.set(style="whitegrid")
g = sns.FacetGrid(df, col='Department', col_wrap=4, height=4, sharey=True)

def plot_actual_forecast(data, **kwargs):
    ax = plt.gca()
    sns.lineplot(data=data, x='Month_Num', y='Actual', marker='o', label='Actual', ax=ax, color='steelblue')
    sns.lineplot(data=data, x='Month_Num', y='Forecast', marker='x', linestyle='--', label='Forecast', ax=ax, color='darkorange')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(range(1, 13))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${int(x/1000)}K'))
    ax.tick_params(labelbottom=True)

g.map_dataframe(plot_actual_forecast)
g.set_titles("{col_name}")
g.set_axis_labels("Month", "Revenue ($)")
g.fig.subplots_adjust(top=0.88, hspace=0.30)
g.fig.suptitle('Actual vs Forecast Revenue by Department (Monthly)', fontsize=16)

# Shared legend
handles, labels = g.axes[0].get_legend_handles_labels()
g.fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.96),
    ncol=2, frameon=False, fontsize='medium'
)

g.savefig("plot1_actual_vs_forecast_by_dept.png", bbox_inches='tight', dpi=300)
plt.close()

# ================================
# Plot 2: Avg % Forecast Deviation by Department
# ================================

plt.figure(figsize=(10, 5))
forecast_error_pct = grouped['pct_variance'].sort_values()
colors = sns.color_palette("RdYlBu_r", len(forecast_error_pct))

bars = sns.barplot(
    x=forecast_error_pct.index,
    y=forecast_error_pct.values,
)

plt.title('Average Forecast Deviation by Department', fontsize=14)
plt.ylabel('Avg Forecast Deviation (%)')
plt.xlabel('Department')
plt.axhline(0, color='gray', linestyle='--')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Add annotations
for bar in bars.patches:
    value = bar.get_height()
    bars.annotate(f'{value:.2f}%', 
                  (bar.get_x() + bar.get_width() / 2, value), 
                  ha='center', va='bottom' if value >= 0 else 'top',
                  fontsize=9)

# Clarify legend
plt.figtext(0.98, 0.02,
            'Positive = Over Forecast\nNegative = Under Forecast',
            fontsize=8, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig("plot2_avg_forecast_deviation_by_dept.png", bbox_inches='tight', dpi=300)
plt.close()

# ================================
# Plot 3: Monthly Actual vs Forecast (All Departments Combined)
# ================================

# Aggregate revenue by month
monthly_summary = df.groupby(['Month_dt'])[['Actual', 'Forecast']].sum().reset_index()
monthly_summary.sort_values(by='Month_dt', inplace=True)
monthly_summary['Month_Label'] = monthly_summary['Month_dt'].dt.strftime('%b-%y')

# Set up the bar plot
plt.figure(figsize=(14, 6))
bar_width = 0.4
x = np.arange(len(monthly_summary))

# Set colors: Actual = blue, Forecast = orange
plt.bar(x - bar_width/2, monthly_summary['Actual'], width=bar_width, label='Actual', color='steelblue')
plt.bar(x + bar_width/2, monthly_summary['Forecast'], width=bar_width, label='Forecast', color='darkorange')

# Titles and labels
plt.title('Monthly Revenue: Actual vs Forecast (All Departments)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Revenue ($M)')
plt.xticks(x, monthly_summary['Month_Label'], rotation=45)

# Format y-axis to millions
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x/1_000_000:.1f}M'))

# Adjust y-axis to fit full range
max_y = max(monthly_summary['Actual'].max(), monthly_summary['Forecast'].max())
plt.ylim(0, max_y * 1.15)

# Add legend
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig("plot3_monthly_actual_vs_forecast.png", bbox_inches='tight', dpi=300)
plt.close()

# ================================
# Plot 4: Smoothed Forecast Accuracy by Department
# ================================

# Sort for rolling
df_sorted = df.sort_values(['Department', 'Month_dt']).copy()

# Rolling 3-month average for smoother view
df_sorted['variance_pct_rolling'] = (
    df_sorted.groupby('Department')['variance_actual_vs_forecast_pct']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

plt.figure(figsize=(13, 6))
palette = sns.color_palette("Paired", df['Department'].nunique())

sns.lineplot(
    data=df_sorted,
    x='Month_dt',
    y='variance_pct_rolling',
    hue='Department',
    palette=palette,
    linewidth=2.0,
    alpha=0.9
)

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Forecast Accuracy Trends by Department (3-Month Avg)', fontsize=14)
plt.ylabel('Variance (%)')
plt.xlabel('Month')
plt.ylim(-0.15, 0.15)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(rotation=45)
plt.legend(title='Department', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig("plot4_smoothed_forecast_accuracy_by_dept.png", bbox_inches='tight', dpi=300)
plt.close()

# ========================================================
# EXECUTIVE SUMMARY — Revenue Forecast Accuracy
# ========================================================
total_actual = df['Actual'].sum()
total_forecast = df['Forecast'].sum()
total_variance = total_actual - total_forecast
total_var_pct = total_variance / total_forecast

lines = [
    "========== Revenue Forecast Executive Summary ==========",
    f"Total Forecast : ${total_forecast:,.0f}",
    f"Total Actual   : ${total_actual:,.0f}",
    f"Total Variance : ${total_variance:,.0f} USD ({total_var_pct:+.2%})",
    ""
]

dept_summary = (
    df.groupby("Department")[["Actual", "Forecast"]]
    .sum()
    .assign(
        Variance = lambda d: d["Actual"] - d["Forecast"],
        Variance_pct = lambda d: d["Variance"] / d["Forecast"]
    )
)

for _, row in dept_summary.iterrows():
    direction = "over" if row["Variance"] > 0 else "under"
    lines.append(
        f"{row.name:<20} {abs(row['Variance_pct'])*100:5.1f}% {direction} forecast  "
        f"(${row['Variance']:>+,.0f} USD)"
    )

# Print and write to file
print("\n".join(lines))

with open("executive_summary.txt", "w") as f:
    f.write("\n".join(lines))

print("\nExecutive summary saved successfully.")
