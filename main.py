import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

url_votes = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-03-23/unvotes.csv"
df_votes = pd.read_csv(url_votes)

vote_mapping = {'yes': 1, 'abstain': 0, 'no': -1}
df_votes['vote_num'] = df_votes['vote'].map(vote_mapping)

matrix = df_votes.pivot_table(index='country', columns='rcid', values='vote_num', fill_value=0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(matrix)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Country'] = matrix.index

plt.figure(figsize=(14, 9))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, color='dodgerblue', edgecolors='w', s=100)

key_countries = [
    'United States', 'Russian Federation', 'China', 'United Kingdom', 'France', 'India', 'Brazil', 'South Africa', 'Germany', 'Spain', 'Mexico', 'Cuba', 'Switzerland', 'Egypt', 'Indonesia'
]

for i, txt in enumerate(pca_df['Country']):
    if txt in key_countries:
        plt.annotate(txt, (pca_df['PC1'].iloc[i] + 0.5, pca_df['PC2'].iloc[i] + 0.5), 
                     fontsize=11, fontweight='bold')

plt.title('UN General Assembly Voting Coalitions (PCA Projection)', fontsize=16, fontweight='bold')
plt.xlabel('First Principal Component (Primary Geopolitical Divide)', fontsize=12)
plt.ylabel('Second Principal Component (Secondary Divide)', fontsize=12)
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()
