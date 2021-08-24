# Title     : plotting_origins.py
# Objective : Produce a clear plot showing the positions of the origins
# Created by: Luke
# Created on: 28/05/2021 11:52

#################################################################
# Python imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#################################################################
# Data imports
data = pd.read_csv(
    r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\pombe_rif1_chip_chrII.csv")
peak_locs = pd.read_csv(
    r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\pombe_rif1_peakcaller.csv")
origin_rif = pd.read_csv(
    r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\pombe_origins_rif1d_chrII.csv")
origin_wt = pd.read_csv(
    r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\pombe_origins_wt_chrII.csv")

#################################################################
# Cleaning dataframes
peak_locs.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
origin_rif.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
origin_wt.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
data.set_index('x', inplace=True)

#################################################################
# Allows specific slices of data to be plotted
start = 100000
end = 120000
#################################################################

# Extract plotting data from dataframes
wt_origin_starts = list(origin_wt['xmin'])
rif_origin_starts = list(origin_rif['xmin'])
wt_origin_ends = list(origin_wt['xmax'])
rif_origin_ends = list(origin_rif['xmax'])
# Origin Efficiency
wt_efficiency = list(origin_wt['efficiency'])
rif_efficiency = list(origin_rif['efficiency'])
# Peaks from MACS peak caller
peak_start = list(peak_locs['start'])
peak_end = list(peak_locs['end'])
peak_enrichment = list(peak_locs['enrichment'])

#################################################################
# Pre-allocate the arrays
origin_wt_locs = [0] * (end - start)
origin_rif_locs = [0] * (end - start)
peakcaller_locs = [0] * (end - start)
# Preparing plotting arrays
datarange = [x for x in range(start, end)]
for wt_start, wt_end, wt_eff in zip(wt_origin_starts, wt_origin_ends, wt_efficiency):
    if wt_start in datarange and wt_end in datarange:
        for coord in range(wt_start - start, wt_end - start):
            origin_wt_locs[coord] = wt_eff
for rif_start, rif_end, rif_eff in zip(rif_origin_starts, rif_origin_ends, rif_efficiency):
    if rif_start in datarange and rif_end in datarange:
        for coord in range(rif_start - start, rif_end - start):
            origin_rif_locs[coord] = rif_eff
for peak_start, peak_end, peak_enrich in zip(peak_start, peak_end, peak_enrichment):
    if peak_start in datarange and peak_end in datarange:
        for coord in range(peak_start - start, peak_end - start):
            peakcaller_locs[coord] = peak_enrich

#################################################################
# Plotly testing
sample = data['rif1'][start:end]
wt_sample = data['mock'][start:end]
norm_sample = data['norm'][start:end]

#################################################################
# Plotting using PlotLy
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=["ChIP Data", "Rif1D Origins",
                                    "Wild Type Origins", "MACS Peaks"],
                    x_title="Chromosome Position")

fig.add_trace(go.Scatter(x=[x for x in range(start, end)], y=sample, name="Rif1 Count", fill='tozeroy',
                         # blue
                         line=dict(color='#1f77b4')), row=1, col=1)

fig.add_trace(go.Scatter(x=[x for x in range(start, end)], y=wt_sample, name="Mock Count", fill='tozeroy',
                         # red
                         line=dict(color='#d62728')), row=1, col=1)

# Below is plotting the norm data
# fig.add_trace(go.Scatter(x=[x for x in range(start, end)], y=norm_sample, name="Norm Count", fill='tozeroy',
#                          # blue
#                          line=dict(color='#1f77b4')), row=1, col=1)

fig.add_trace(go.Scatter(x=[x for x in range(start, end)], y=origin_rif_locs, name="Rif1 Origins Efficiency",
                         # blue
                         line=dict(color='#1f77b4')), row=2, col=1)

fig.add_trace(go.Scatter(x=[x for x in range(start, end)], y=origin_wt_locs, name="WT Origins Efficiency",
                         # red
                         line=dict(color='#d62728')), row=3, col=1)

fig.add_trace(go.Scatter(x=[x for x in range(start, end)], y=peakcaller_locs, name="Peak Enrichment",
                         # red
                         line=dict(color='#2ca02c')), row=4, col=1)

#################################################################
# Figure updating. Labelling etc
fig['layout']['yaxis1'].update(title='Count')
# 'Kanoh Y, et al. Rif1 binds to G quadruplexes and suppresses replication over long distances. (2015)'
fig['layout']['yaxis2'].update(title='Efficiency', range=[
                               0.0, max(rif_efficiency, wt_efficiency)])
fig['layout']['yaxis3'].update(title='Efficiency', range=[
                               0.0, max(rif_efficiency, wt_efficiency)])
fig['layout']['yaxis4'].update(title='Enrichment', range=[
                               0.0, max(peak_enrichment)])

fig.update_traces(mode="lines", hovertemplate=None)
fig.update_layout(
    title="Plot of Pombe ChIP Data with Origin Subplots",
    font=dict(family="Times New Roman, monospace", size=14),
    hovermode="x unified")
# This stops vertical panning when in browser
fig.update_yaxes(fixedrange=True)

# print(start + 20)
#################################################################
# Save the plot as an .HTML
# fig.write_html(r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\chip_plot.html")
# fig.write_html(r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\norm_chip_plot.html")


fig.show()
