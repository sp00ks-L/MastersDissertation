# Title      :	predictions with origins.py
# Objective  :	Create a Plotly webplot to show my model's predictions alongside information about the DNA origins
# Created by :	Luke
# Created on :	Thu 19/08/21 11:59

#################################################################
# Python imports
import numpy as np

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#################################################################
# Loading Data
chip_data = pd.read_csv(
    rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_rif1_chip_chrI.csv")
peak_data = pd.read_csv(
    r'C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_rif1_peakcaller_chrI.csv')
origin_rif = pd.read_csv(
    r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_origins_rif1d_chrI.csv")
origin_wt = pd.read_csv(
    r"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_origins_wt_chrI.csv")

#################################################################
# Load in predictions and labels
dir = r'E:\CNN Models\Predictions'
label_preds = pd.read_csv(dir + r'\labels.csv')
label_probs = pd.read_csv(dir + r'\confidence.csv')
threshold_loc = r'E:\CNN Models\best_threshold.txt'
f = open(threshold_loc, "r")
threshold = float(f.read())
f.close()

#################################################################
# Conversion of bool to 0 | 1
label_preds_list = list(label_preds['binding_site'].astype(int))
# Find probability labels that exceed the threshold
label_probs_list = [float(prob)
                    for prob in label_probs['mean'] if prob > threshold]

# Parameters for Plotting and Predictions
slice_size = 110000
# the window size will depend on the size of the prediction images
window_size = 5000
jump_size = window_size // 2

# Extract origin location from dataframes
wt_origin_starts = list(origin_wt['xmin'])
rif_origin_starts = list(origin_rif['xmin'])
wt_origin_ends = list(origin_wt['xmax'])
rif_origin_ends = list(origin_rif['xmax'])
# Origin Efficiency
wt_efficiency = list(origin_wt['efficiency'])
rif_efficiency = list(origin_rif['efficiency'])

#################################################################
# Pre-allocate the arrays
origin_wt_locs = [0] * slice_size
origin_rif_locs = [0] * slice_size
# Preparing plotting arrays
datarange = [x for x in range(slice_size)]
for wt_start, wt_end, wt_eff in zip(wt_origin_starts, wt_origin_ends, wt_efficiency):
    # For each WT origin in the datarange
    if wt_start in datarange and wt_end in datarange:
        for coord in range(wt_start, wt_end):
            # For each x_coord in the origin
            origin_wt_locs[coord] = wt_eff
for rif_start, rif_end, rif_eff in zip(rif_origin_starts, rif_origin_ends, rif_efficiency):
    if rif_start in datarange and rif_end in datarange:
        for coord in range(rif_start, rif_end):
            origin_rif_locs[coord] = rif_eff

origin_eff_difference = np.array(
    [x - y for x, y in zip(origin_rif_locs, origin_wt_locs)])

# Finding start index for each binding prediction region
binding_regions = []
binding_img_indices = np.where(np.array(label_preds_list) == 1)[0]

"""
Multiply the indices by the jump size; this is required due to the way that the prediction images
are made. If we start at 0-5000 for the 1st img, the 2nd img will be 2500-7500 because jump_size = window // 2
This means that the indices that we extract of binding images need to be multiplied by jump_size so they correspond
correctly to their position on the chromosome - this gives us the starting index on the chromosome for each binding
region
"""
binding_img_indices *= jump_size

# Expand binding regions to encompass the 5kb window
for start_ind in binding_img_indices:
    if start_ind >= slice_size:
        break
    end_ind = start_ind + window_size
    if end_ind <= slice_size:  # if binding region fits within data slice
        binding_regions.append([start_ind, end_ind])
    else:
        # if binding region doesnt fit in slice, we need to clip the region
        binding_regions.append([start_ind, slice_size])

# Find the peak regions from the peak-caller
peaks = []
for i in range(len(peak_data)):
    peak_start = peak_data['start'][i]
    peak_end = peak_data['end'][i]
    if peak_data['start'][i] >= slice_size:
        break
    if peak_start < slice_size and peak_end <= slice_size:
        peaks.append([peak_start, peak_end])
    else:
        peaks.append([peak_start, slice_size])

#################################################################
# Here I want to filter only the binding predictions that are near or within KNOWN origin
# This is only required for the Plotly filter to work
filtered_binding_locs = []
# if prediction near an origin say (+- 2.5kb) also include it
for bs in binding_regions:
    x1 = bs[0] - 2500
    x2 = bs[1] + 2500
    tmp_slice = origin_eff_difference[x1:x2]
    if max(tmp_slice) >= 0.2:
        filtered_binding_locs.append(bs)

# I need a bool array to filter the plotly traces
filtered_bool = [
    True if x in filtered_binding_locs else False for x in binding_regions]
#################################################################
# Plotting the predictions and peaks using plotly
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=[
                        "<b>Original Peaks and Model Predictions</b>",
                        "<b>Difference in Origin Efficiency between Wild-type (WT) and <i>rif1</i></b>" +
                        u'<b>\u0394</b>',
                        "<b>Origin Locations</b>"],
                    x_title="Chromosome Position")

fig.add_trace(go.Scatter(x=[x for x in range(0, slice_size)], y=chip_data['norm'][:slice_size], name="ChIP Data",
                         line=dict(color='#c44e52')), row=1, col=1)  # red

# To ensure multiple legend labels are not added for each peak, use the sentinel label_added
label_added = False
for peak in peaks:
    # peaks are blue
    if not label_added:
        fig.add_trace(go.Scatter(
            x=[x for x in range(peak[0], peak[1])], y=chip_data['norm'][peak[0]:peak[1]], name='Original Peak',
            fill='tozeroy', line=dict(color='#1f77b4'), showlegend=True), row=1, col=1)
        label_added = True
    else:
        fig.add_trace(go.Scatter(
            x=[x for x in range(peak[0], peak[1])], y=chip_data['norm'][peak[0]:peak[1]], name='Original Peak',
            fill='tozeroy', line=dict(color='#1f77b4'), showlegend=False), row=1, col=1)

label_added = False
conf_pct_levels = []
max_dist = max([float(_) for _ in label_probs['mean']]) - threshold
for i in range(len(binding_regions)):
    conf_level = label_probs_list[i] - threshold
    # finding the confidence level as a percentage (or fraction) of the maximum
    conf_level_pct = conf_level / max_dist
    conf_pct_levels.append(conf_level_pct)

    # Just to shorten following line
    x1 = binding_regions[i][0]
    x2 = binding_regions[i][1]
    # Predictions are green - fill has 50% alpha
    fill_col = f'rgba(85, 168, 104, 0.5)'
    if not label_added:
        fig.add_trace(go.Scatter(x=[x for x in range(x1, x2)], y=chip_data['norm'][x1:x2], name='Binding Prediction',
                                 fill='tozeroy', fillcolor=fill_col, line=dict(
            color='#55a868'), showlegend=True, text=f'Conf % {round(conf_level_pct * 100, 3)}'), row=1, col=1)
        label_added = True
    else:
        fig.add_trace(go.Scatter(x=[x for x in range(x1, x2)], y=chip_data['norm'][x1:x2], name='Binding Prediction',
                                 fill='tozeroy', fillcolor=fill_col, line=dict(
            color=f'rgba(85, 168, 104, {conf_level_pct})'), showlegend=False,
            text=f'Conf % {round(conf_level_pct * 100, 3)}'), row=1, col=1)

#################################################################
# Plotting origin efficiency difference
# u'-\u0394' is unicode for capital Delta
fig.add_trace(
    go.Scatter(x=[x for x in range(0, slice_size)], y=np.where(origin_eff_difference < 0, origin_eff_difference, 0),
               name=u'-\u0394' + 'Efficiency', fill='tozeroy', line=dict(color='red'), showlegend=False), row=2, col=1)

fig.add_trace(
    go.Scatter(x=[x for x in range(0, slice_size)], y=np.where(origin_eff_difference > 0, origin_eff_difference, 0),
               name=u'+\u0394' + 'Efficiency', fill='tozeroy', line=dict(color='green'), showlegend=False), row=2,
    col=1)
fig.add_trace(go.Scatter(x=[x for x in range(0, slice_size)], y=[0 for _ in range(0, slice_size)], name=u'\u0394' +
                                                                                                        'Efficiency',
                         line=dict(color='black'), showlegend=False), row=2, col=1)
#################################################################
# Plotting Origin Locations
fig.add_trace(go.Scatter(x=[x for x in range(0, slice_size)], y=origin_wt_locs, name='WT Origins', fill='tozeroy',
                         # yellow colour
                         line=dict(color='#ccb974'), showlegend=True), row=3, col=1)
fig.add_trace(
    go.Scatter(x=[x for x in range(0, slice_size)], y=origin_rif_locs, name='<i>rif1</i>' + u'\u0394' + ' Origins',
               fill='tozeroy',
               # purple colour
               line=dict(color='#8172b3'), showlegend=True), row=3, col=1)

bool_list = [True] * len(fig['data'])

""""
Each plotly trace has its own index, when plotting in a for loop, the number of indices can be hard to track
plot_number is a count of the number of peaks already plotted because each individial peak is its own plotly trace
the + 1 takes into account that we have already plotted the chip data which again, is its own trace
"""
plot_number = len(peaks) + 1
# This is the bool list for filtering the plotly trace
bool_list[plot_number:len(filtered_bool)] = filtered_bool
#################################################################
# Adding Plot Filters with Dropdown
fig.update_layout(
    updatemenus=[go.layout.Updatemenu(
        active=0,
        bgcolor='#ebebeb',
        buttons=list(
            [dict(label='All Predictions',
                  method='update',
                  args=[{'visible': [True] * len(fig['data'])},
                        {'title': "Model Predictions on Chromosome I of <i>S. pombe</i> with Origin Subplots",
                         'showlegend': True}]),
             dict(label='Predictions Near Origins',
                  method='update',
                  args=[{'visible': bool_list},  # the index of True aligns with the indices of plot traces
                        {'title': "Model Predictions on Chromosome I of <i>S. pombe</i> with Origin Subplots",
                         'showlegend': True}]),
             dict(label='Toggle Confidence Levels',
                  method='restyle',
                  args=[{'fillcolor': [f'rgba(85, 168, 104, {i})' for i in conf_pct_levels]}, [_ for _ in range(
                      plot_number, plot_number + len(filtered_bool))]],
                  args2=[{'fillcolor': f'rgba(85, 168, 104, 0.5)'}, [_ for _ in range(
                      plot_number, plot_number + len(filtered_bool))]])  # has 2 args because its a toggle button
             ]),
        direction='right',
        pad={'r': 0, 't': 0, 'b': 10},
        type='buttons',
        showactive=True,
        font={'size': 12},
        x=1.3,
        xanchor='right',
        y=1.05,
        yanchor='top'
    )
    ])

#################################################################
# Figure updating. Labelling etc
fig.update_layout(
    title="Model Predictions on Chromosome I of <i>S. pombe</i> with Origin Subplots",
    font=dict(size=22))

# this controls the subplot title and the x label
fig.update_annotations(font_size=20)
fig['layout']['yaxis1'].update(title='Normalised Count')
# 'Kanoh Y, et al. Rif1 binds to G quadruplexes and suppresses replication over long distances. (2015)'
fig['layout']['yaxis2'].update(title=u'\u0394' + 'Origin Efficiency')
fig['layout']['yaxis3'].update(title='Origin Efficiency', range=[
                               0.0, max(rif_efficiency, wt_efficiency)])
fig.for_each_yaxis(lambda axis: axis.title.update(font=dict(size=20)))
fig.update_xaxes(tickfont_size=18, ticks="outside")
fig.update_yaxes(tickfont_size=18, ticks="outside")

# prevents vertical panning on plotly which isnt required here
fig.update_yaxes(fixedrange=True)
# fig.write_html(
#     r"C:\Users\Luke\Sussex Code\Personal Code\chip_predictions.html")


fig.show()
