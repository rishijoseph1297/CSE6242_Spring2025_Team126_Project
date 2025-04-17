# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# vis_dash_cse6242
# http://localhost:8501/
# #!pip install streamlit (uncomment to install streamlit or run "pip install streamlit in terminal)
# once installed run "streamlit run vis_dash_cse6242.py" in terminal

# + editable=true slideshow={"slide_type": ""}
### Cell Block for Initial Layout, Player Selct, Image Comparisons, Top 5 Table and Sliders
import os
import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches

# --- Helper Functions ---
def get_nba_pic(query):
    return f"https://source.unsplash.com/featured/?{urllib.parse.quote(query)}"

def in_2_ft_in(inches):
    try:
        inches = float(inches)
        if inches > 96:
            inches = inches * 0.393701
        if inches < 60:
            return "6'0\""
        feet = int(inches // 12)
        inch = int(round(inches % 12))
        return f"{feet}'{inch}\""
    except:
        return "6'0\""

def normalize_height(height):
    try:
        h = float(height)
        if h > 96:
            h = h * 0.393701
        if h < 60:
            return 72
        return h
    except:
        return 72

# --- Load Data ---
college_df = pd.read_csv("player_data_college_latest_season.csv")
nba_df = pd.read_csv("nba_player_avgs_2008-2025.csv")
map_df = pd.read_csv("nba_ncaa_map.csv")

college_df['height'] = college_df['height'].apply(normalize_height)
comp_pool_df = college_df.copy()
comp_pool_df['height'] = comp_pool_df['height'].apply(normalize_height)
nba_df['height'] = nba_df['height'].apply(normalize_height)

college_df['height_ft_in'] = college_df['height'].apply(in_2_ft_in)
nba_df['height_ft_in'] = nba_df['height'].apply(in_2_ft_in)

prospects_df = college_df[college_df['max_year'] == 2025].copy()
comp_pool_df = college_df[college_df['ncaa_id'].isin(map_df['ncaa_id'])].copy()
comp_pool_df = comp_pool_df.sort_values(by="points_scored", ascending=False)
comp_pool_df = comp_pool_df.drop_duplicates(subset="ncaa_id", keep="first")

player_options = prospects_df[['player_name', 'ncaa_id']].drop_duplicates()
player_dict = dict(zip(player_options['player_name'], player_options['ncaa_id']))

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
nba_logo_url = "https://1000logos.net/wp-content/uploads/2017/04/nba-big-logo.png"

st.markdown(
    f"""
    <div style='display: flex; justify-content: left; align-items: center; gap: 12px; margin-bottom: 20px;'>
        <img src="{nba_logo_url}" width="150">
        <h1 style="font-size:3.2rem;">Draft Prospect Comparison Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# st.title("🏀 NBA Draft Prospect Comparison Dashboard")

selected_player_name = st.selectbox("Select a 2024–2025 College Player", sorted(player_dict.keys()))

if selected_player_name:
    ncaa_id = player_dict[selected_player_name]
    player_row = prospects_df[prospects_df['ncaa_id'] == ncaa_id].iloc[0]

    # ESPN image match
    espn_map_df = pd.read_csv("espn_ncaa_player_ids.csv") if os.path.exists("espn_ncaa_player_ids.csv") else pd.DataFrame()
    espn_map_df.columns = espn_map_df.columns.astype(str).str.strip().str.lower()
    search_name = selected_player_name.lower().replace('.', '').replace(' ', '')
    candidate_names = espn_map_df['player_name'].str.lower().str.replace('.', '', regex=False).str.replace(' ', '', regex=False)
    matched_name = get_close_matches(search_name, candidate_names.tolist(), n=1, cutoff=0.6)
    espn_id_row = espn_map_df.loc[[candidate_names[candidate_names == matched_name[0]].index[0]]] if matched_name else pd.DataFrame()

    if selected_player_name == "Dylan Harper":
        college_img = "https://www.proballers.com/media/cache/resize_600_png/https---www.proballers.com/ul/player/dylan-harper-1ef9e125-e048-67ec-82bf-d3649d8ce32c.png"
    elif selected_player_name == "Ace Bailey":
        college_img = "https://www.usab.com/imgproxy/w0x0S1qSu8FakrZ8B1tPs3BVFsvwtSvwRsx-cbmtYEg/rs:fit:3000:0:0:g:ce/aHR0cHM6Ly9zdG9yYWdlLmdvb2dsZWFwaXMuY29tL3VzYWItY29tLXByb2QvdXBsb2FkLzIwMjQvMDQvMTEvMjRkODQ4NWItNjRjMi00ZTc5LWIxNzAtNjllNTJkN2I1ZDJlLmpwZw.png"
    elif not espn_id_row.empty:
        espn_id = espn_id_row.iloc[0]['espn_id']
        college_img = f"https://a.espncdn.com/combiner/i?img=/i/headshots/mens-college-basketball/players/full/{espn_id}.png"
    else:
        college_img = "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png"

    # Feature engineering
    comp_pool_df['ft_pct'] = comp_pool_df['freethrow_made'] / comp_pool_df['freethrow_attempted'].replace(0, np.nan)
    comp_pool_df['2p_pct'] = comp_pool_df['2p_made'] / comp_pool_df['2p_attempted'].replace(0, np.nan)
    comp_pool_df['3p_pct'] = comp_pool_df['3p_made'] / comp_pool_df['3p_attempted'].replace(0, np.nan)
    comp_pool_df['tov'] = comp_pool_df['tov_per_game'] if 'tov_per_game' in comp_pool_df.columns else 0
    comp_pool_df['def_rating'] = comp_pool_df['defensive_rating'] if 'defensive_rating' in comp_pool_df.columns else 0

    features = ["points_scored", "assists", "total_reb", "blocks", "steals", "height", "ft_pct", "2p_pct", "3p_pct", "tov", "def_rating"]
    X = comp_pool_df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -- MAIN LAYOUT --
    left_col, right_col = st.columns([1.1, 2])

    # Sliders on left
    with left_col:
        header_col, button_col = st.columns([5, 1])
        stat_map = {
            "Points": "points_scored",
            "Assists": "assists",
            "Rebounds": "total_reb",
            "Blocks": "blocks",
            "Steals": "steals"
        }
        with header_col:
            st.markdown("#### 🔧 Adjust Player Stats")
        with button_col:
            if st.button("🔁", help="Reset to original stats"):
                for label, col in stat_map.items():
                    st.session_state[f"slider_{label.lower()}"] = float(player_row[col])

        
        # Label-to-column mapping

        if "last_player" not in st.session_state or st.session_state["last_player"] != selected_player_name:
            st.session_state["last_player"] = selected_player_name
            selected_player_row = prospects_df[prospects_df["player_name"] == selected_player_name].iloc[0]
            for label, col in stat_map.items():
                st.session_state[f"slider_{label.lower()}"] = float(selected_player_row[col])
        
        # Sliders using session state
        adjusted_stats = {
            label: st.slider(
                label,
                0.0,
                30.0 if label == "Points" else 20.0 if label in ["Assists", "Rebounds"] else 10.0,
                st.session_state.get(f"slider_{label.lower()}", float(player_row[col])),
                0.1,
                key=f"slider_{label.lower()}"
            )
            for label, col in stat_map.items()
        }

    # Images + Table on right
    with right_col:
        img1, img2 = st.columns(2)
        with img1:
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src="{college_img}" width="200"><br>
                    <strong>Draft Prospect</strong><br>
                    {selected_player_name}<br>
                    Height: {player_row['height_ft_in']}
                </div>
                """, unsafe_allow_html=True
            )
        # --- k-NN Logic ---
        adjusted_vector = np.array([
            adjusted_stats["Points"], adjusted_stats["Assists"], adjusted_stats["Rebounds"],
            adjusted_stats["Blocks"], adjusted_stats["Steals"], player_row['height'],
            player_row['freethrow_made'] / player_row['freethrow_attempted'] if player_row['freethrow_attempted'] else 0,
            player_row['2p_made'] / player_row['2p_attempted'] if player_row['2p_attempted'] else 0,
            player_row['3p_made'] / player_row['3p_attempted'] if player_row['3p_attempted'] else 0,
            player_row['tov_per_game'] if 'tov_per_game' in player_row else 0,
            player_row['defensive_rating'] if 'defensive_rating' in player_row else 0
        ]).reshape(1, -1)

        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(scaler.transform(adjusted_vector))
        comp_results = comp_pool_df.iloc[indices[0]].copy()

        top_comp_name = comp_results.iloc[0]['player_name']
        top_comp_ncaa_id = comp_results.iloc[0]['ncaa_id']
        top_nba_id_row = map_df[map_df['ncaa_id'] == top_comp_ncaa_id]
        if not top_nba_id_row.empty:
            top_nba_id = top_nba_id_row.iloc[0]['nba_id']
            top_comp_img = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{top_nba_id}.png"
        else:
            top_comp_img = get_nba_pic(f"{top_comp_name} NBA headshot")

        with img2:
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src="{top_comp_img}" width="200"><br>
                    <strong>Top NBA Comp</strong><br>
                    {top_comp_name}<br>
                    Height: {comp_results.iloc[0]['height_ft_in']}
                </div>
                """, unsafe_allow_html=True
            )

        html = """
        <style>
            .comparison-table {
                width: 90%;
                border-collapse: separate;
                border-spacing: 0;
                margin: auto;
                text-align: center;
                border-radius: 8px;
                overflow: hidden;
            }
            .comparison-table th {
                background-color: #2d2f33;
                color: white;
                padding: 5px 10px;
                font-weight: 600;
            }
            .comparison-table td {
                color: white;
                padding: 1px;
            }
        </style>
        
        <h4 style='text-align: center; margin-top: 30px;'>🥇Top 5 NBA Comparisons</h4>
        
        <table class='comparison-table'>
            <tr><th>Rank</th><th>Name</th></tr>
        """
        
        for i, name in enumerate(comp_results['player_name'], start=1):
            html += f"<tr><td>{i}</td><td>{name}</td></tr>"
        
        html += "</table>"
        
        st.markdown(html, unsafe_allow_html=True)

# +
### Nick S' Code For PCA Player Type 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, euclidean_distances
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from collections import defaultdict

import numpy as np
np.random.seed(6242)

###################################
######## Data Input ###############
###################################

# read in data
players_df_edited = pd.read_csv('player_data_college_latest_season.csv')

# apply filters
players_df_edited.dropna(inplace=True)

only_NBA = True
if only_NBA:
    nba_map = pd.read_csv('nba_ncaa_map.csv')
    players_df_edited = players_df_edited[players_df_edited['ncaa_id'].isin(nba_map['ncaa_id'])]

# remove more columns for model processing
model_columns_to_drop = ['player_name', 'team', 'min_year', 'max_year', 'pos_class', 'type', 'avg_pick', 'ncaa_id']
players_df_model = players_df_edited.drop(
    columns=[col for col in model_columns_to_drop if col in players_df_edited.columns])


# clean data (remove NaN)


###################################
######## Feature Selection ########
###################################

# standardize data
scaler = StandardScaler()
scaler_players_df = scaler.fit_transform(players_df_model)



def principal_component_selection(X, n_clusters=None, n_components=10):
    # feature selection kmeans based on PCA, structure from
    # https://datascience.stackexchange.com/questions/67040/how-to-do-feature-selection-for-clustering-and-implement-it
    # -in-python
    pca = PCA(n_components=n_components).fit(X)
    A_q = pca.components_.T

    if n_clusters == None:
        kmeans = KMeans(n_init='auto').fit(A_q)
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(A_q)
    clusters = kmeans.predict(A_q)
    cluster_centers = kmeans.cluster_centers_

    dists = defaultdict(list)
    for i, c in enumerate(clusters):
        dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
        dists[c].append((i, dist))

    # gives inds of selected features based on variance
    pfa_indices = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]

    return pfa_indices


# pfa feature selected model input
pfa_inds_23_24 = principal_component_selection(scaler_players_df)
pfa_players_df = players_df_model.iloc[:, pfa_inds_23_24]
print(pfa_players_df.columns)

# hand selected model input; advanced shooting data only
hand_selected_vars = ['offensive_rebound_pct', 'defensive_rebound_pct', '2p_made']
hand_players_df = players_df_model[hand_selected_vars]


###################################
######## Feature Evaluation #######
###################################

# big loop to compare cluster difference scores
def run_k_selection_and_record_scores(datasets, k_range=(2, 11)):
    results = []

    for dataset_name, data in datasets.items():
        print(f"Processing dataset: {dataset_name}")

        # Initialize KMeans and KElbowVisualizer
        model = KMeans(random_state=1, n_init='auto')
        visualizer = KElbowVisualizer(model, k=k_range, metric='calinski_harabasz', timings=False, locate_elbow=True)

        # Fit the visualizer to the data
        visualizer.fit(data)
        # Record silhouette scores
        for k, score in zip(range(k_range[0], k_range[1] + 1), visualizer.k_scores_):
            results.append({'Dataset': dataset_name, 'K': k, 'CH Index': score})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# list of datasets to compare
datasets = {
    'pfa_players_df': StandardScaler().fit_transform(pfa_players_df),
    #'hand_players_df': StandardScaler().fit_transform(hand_players_df),
    'players_df_model': StandardScaler().fit_transform(players_df_model),
}

# execute
k_range = (3, 15)
results_df = run_k_selection_and_record_scores(datasets, k_range=k_range)
best_CH_row = best_row = results_df.loc[results_df["CH Index"].idxmax()]
best_dataset = best_row["Dataset"]
best_dataset_obj = datasets[best_dataset]
best_k = best_row["K"]
results_df.to_csv('eval_scores.csv', index=False)
plt.close()

####################################
# Final KMeans Implementation ######
####################################

####################################
# Final KMeans Implementation ######
####################################

# run KMeans on the best dataset
kmeans_model_23_24 = KMeans(random_state=1, n_init='auto', n_clusters=best_k)
kmeans_model_23_24.fit(best_dataset_obj)

# predict clusters
players_df_edited = players_df_edited.reset_index(drop=True)
players_df_edited['cluster'] = kmeans_model_23_24.predict(best_dataset_obj)

####################################
########### PCA and graph ##########
####################################

# Run PCA for 2 components
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(best_dataset_obj)
pca_df = pd.DataFrame(pca_transformed, columns=['PC1', 'PC2'])

# Reset index to ensure alignment
pca_df = pca_df.reset_index(drop=True)
players_df_edited['PC1'] = pca_df['PC1']
players_df_edited['PC2'] = pca_df['PC2']

# Save results
players_df_edited.to_csv('player_cluster.csv', index=False)

####################################
####### Positional Purity ##########
####################################
# find positional mode of each cluster
positional_purity = True
if positional_purity:
    def positional_purity(player_df):
        # Step 1: Get most likely raw position
        position_stats = (
            player_df.groupby('cluster')['pos_class']
            .apply(lambda x: x.mode()[0])
            .reset_index(name='most_likely_position')
        )
    
        # Step 2: Calculate mode percentage
        position_percentage = (
            player_df.groupby('cluster')['pos_class']
            .apply(lambda x: (x == x.mode()[0]).mean() * 100)
            .reset_index(name='position_percentage')
        )
    
        # Step 3: Merge both
        cluster_position_info = pd.merge(position_stats, position_percentage, on='cluster')
    
        # Step 4: Remap to simplified categories
        position_remap = {
            'Wing G': 'Wing/G',
            'Wing F': 'SF',
            'Pure PG': 'PG',
            'PG': 'Wing/G',
            'SG': 'Wing/G',
            'SF': 'SF',
            'C': 'C',
            'PF': 'PF/C',
            'PF/C': 'PF/C',
        }
        cluster_position_info['position_category'] = cluster_position_info['most_likely_position'].replace(position_remap)
    
        # Step 5: Enforce uniqueness (e.g., "Wing/G", "Wing/G (1)", etc.)
        used_labels = {}
        unique_labels = []
        for label in cluster_position_info['position_category']:
            if label not in used_labels:
                used_labels[label] = 1
                unique_labels.append(label)
            else:
                new_label = f"{label} ({used_labels[label]})"
                used_labels[label] += 1
                unique_labels.append(new_label)
    
        cluster_position_info['unique_position_label'] = unique_labels
    
        return cluster_position_info



    clusters_df = positional_purity(players_df_edited)
    clusters_df.to_csv('cluster_stats.csv', index=False)

# + editable=true slideshow={"slide_type": ""}
# Plotting PCA Player Type Cluster Graph, Dyanmic Player Movement in Graph w/ Slider Adjustments
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import streamlit as st

# Re-load the data since the previous variable definitions may be out of scope
player_cluster_df = pd.read_csv("player_cluster.csv")
cluster_stats_df = pd.read_csv("cluster_stats.csv")

# Prepare data
pca_data = player_cluster_df[['PC1', 'PC2', 'cluster']]
positions = cluster_stats_df.set_index('cluster')['unique_position_label'].to_dict()

# Create enhanced PCA cluster plot

# Plot each cluster
unique_clusters = sorted(pca_data['cluster'].unique())
# colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
cluster_colors = {
    0: {'point': '#d3d3d3', 'fill': '#d3d3d3', 'edge': '#a9a9a9'},  # Light gray
    1: {'point': '#add8e6', 'fill': '#add8e6', 'edge': '#5f9ea0'},  # Light blue
    2: {'point': '#f08080', 'fill': '#f08080', 'edge': '#cd5c5c'},  # Light red
}
fig, ax = plt.subplots(figsize=(10, 6), facecolor='none', dpi=500)

for cluster in unique_clusters:
    cluster_data = pca_data[pca_data['cluster'] == cluster][['PC1', 'PC2']].values
    position_label = positions.get(cluster, "N/A")
    
    # Get color scheme
    color_scheme = cluster_colors.get(cluster, {'point': 'gray', 'fill': 'gray', 'edge': 'black'})
    
    # Scatter points
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
               label=f'{position_label}', 
               color=color_scheme['point'], 
               alpha=0.6, s=5)

    # Plot convex hull
    if len(cluster_data) >= 3:
        hull = ConvexHull(cluster_data)
        hull_vertices = cluster_data[hull.vertices]
        
        # Fill hull
        ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], 
                color=color_scheme['fill'], 
                alpha=0.15, zorder=1)
        
        # Draw hull outline
        for simplex in hull.simplices:
            ax.plot(cluster_data[simplex, 0], cluster_data[simplex, 1], 
                    color=color_scheme['edge'], 
                    linewidth=1, zorder=2)



    # Annotate with most likely position
    centroid = cluster_data.mean(axis=0)
    position_label = positions.get(cluster, "N/A")
    ax.text(centroid[0], centroid[1], position_label, fontsize=15, fontweight='bold', ha='center', va='center',
            color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

# Styling
ax.set_facecolor("none")
ax.grid(True, color="gray", linestyle="--", alpha=0.3)

# Set text colors to white
ax.set_xlabel("PC1", labelpad=10, color="white", fontsize=12)
ax.set_ylabel("PC2", labelpad=10, color="white", fontsize=12)
ax.set_title("Player Type Projection - PCA Clusters", pad=15, color="white", fontsize=19, fontweight='bold')

# Set tick labels to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Set legend text to white
legend = ax.legend(loc="best", fontsize=12, markerscale=3)
for text in legend.get_texts():
    text.set_color("white")

# PCA Cluster Player Marker
players_df_edited = pd.read_csv('player_data_college_latest_season.csv')

# apply filters
player_row_df = players_df_edited[players_df_edited['ncaa_id'] == ncaa_id]

model_columns_to_drop = ['player_name', 'team', 'min_year', 'max_year', 'pos_class', 'type', 'avg_pick', 'ncaa_id']
player_row_df = player_row_df.drop(
     columns=[col for col in model_columns_to_drop if col in player_row_df.columns])
player_row_df = player_row_df.fillna(0)
# Map slider labels to column names in player_row_df
slider_to_col = {
    "Points": "points_scored",
    "Assists": "assists",
    "Rebounds": "total_reb",
    "Blocks": "blocks",
    "Steals": "steals"
}

# Update player_row_df with adjusted slider values
for slider_label, col_name in slider_to_col.items():
        player_row_df[col_name] = adjusted_stats[slider_label]
single_row_scaled = scaler.transform(player_row_df)
pc1_pc2 = pca.transform(single_row_scaled)
pc1, pc2 = pc1_pc2[0]

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

basketball_img = mpimg.imread("basketball.png")
imagebox = OffsetImage(basketball_img, zoom=0.025)  # Adjust zoom if needed
ab = AnnotationBbox(imagebox, (pc1, pc2), frameon=False, zorder=10)
ax.add_artist(ab)
ax.text(pc1 + 0.3, pc2, selected_player_name,
        fontsize=10, color='black', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.2'))

# Show plot
# %matplotlib inline
col1, col2 = st.columns([1, 1.8])
with col2:
    st.pyplot(fig, clear_figure=True)


# + editable=true slideshow={"slide_type": ""}
### Creating Projection Combo Plots (Bar Chart w/ Scatterplots) Comparing Selected Player Projections
### Comparisons to League averages in those same years (Bar Chart)
### Filter Feature to View Projections for Points, Assits, Rebounds or All at the Same time

# --- Helper: Get Player First 5 Seasons (by top_nba_id) ---
def get_player_first_5_seasons_by_id(nba_id, stat_col):
    player_df = nba_df[nba_df['personId'] == nba_id].sort_values(by='season').copy()
    player_df = player_df.head(5)
    league_avg = nba_df[nba_df['season'].isin(player_df['season'])].groupby('season')[stat_col].mean().reset_index()
    return player_df[['season', stat_col]], league_avg

# --- Plot Function ---
def plot_player_vs_league(player_data, league_data, stat, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Create numeric 'Year' index (up to 5)
    player_data = player_data.copy()
    player_data['Year'] = range(1, len(player_data) + 1)

    league_data = league_data.copy()
    league_data['Year'] = range(1, len(league_data) + 1)

    # Player line plot
    ax.plot(
        player_data['Year'],
        player_data[stat],
        marker='o',
        linewidth=2.5,
        color='#66b3ff',
        label='Player'
    )

    # League average bar plot
    ax.bar(
        league_data['Year'],
        league_data[stat],
        alpha=0.3,
        color='#d3d3d3',
        label='League Avg'
    )

    # Transparent background, clean styling
    ax.set_facecolor('none')
    ax.figure.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Axis & tick styling
    # ax.set_title(f"{stat_labels[stat]} Over First {len(player_data)} Seasons", fontsize=12, color='white')
    ax.set_xlabel("Year", color='white')
    ax.set_ylabel(stat_labels[stat], color='white')
    ax.tick_params(colors='white')
    ax.set_xticks(player_data['Year'])  # dynamic x-axis ticks

    ax.legend()
    legend = ax.legend(loc="best", fontsize=12, markerscale=1)
    for text in legend.get_texts():
        text.set_color('white')
    return ax

# --- Stat Projections UI and Layout ---

stat_options = ['points', 'assists', 'reboundsTotal', 'numMinutes']
stat_labels = {
    'points': 'Points',
    'assists': 'Assists',
    'reboundsTotal': 'Rebounds',
    'numMinutes': 'Minutes'
}
# selected_stat = st.selectbox("Select a stat to view:", stat_options + ['ALL'])

# Layout columns (you already have col1 and col2 defined)
with col1:
    st.markdown("#### 📈Stat Projections")
    selected_stat = st.selectbox(
        "Select a stat to view:",
        stat_options + ['ALL'],
        format_func=lambda x: stat_labels.get(x, x)  # fallback to key if not found
    )
    if 'top_nba_id' in locals() or 'top_nba_id' in globals():
    #with col1:
        if selected_stat != 'ALL':
            player_data, league_data = get_player_first_5_seasons_by_id(top_nba_id, selected_stat)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            plot_player_vs_league(player_data, league_data, selected_stat, ax)
            st.pyplot(fig, clear_figure=True)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for i, stat in enumerate(stat_options):
                player_data, league_data = get_player_first_5_seasons_by_id(top_nba_id, stat)
                plot_player_vs_league(player_data, league_data, stat, axes[i // 2, i % 2])
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
    else:
        #with col1:
        st.info("Please select a college player to view their top NBA comp and projections.")

# -


