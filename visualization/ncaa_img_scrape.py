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
# # !pip install selenium (if needed)
# Scraping NCAA image ID's for image comparisons in vis_dash_cse6242.py
# Export ID's to espn_ncaa_player_ids.csv

# Setup headless browser
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

def get_all_team_ids():
    url = "https://www.espn.com/mens-college-basketball/teams"
    driver.get(url)
    time.sleep(2)

    seen = set()
    teams = []
    links = driver.find_elements(By.CSS_SELECTOR, 'a.AnchorLink')
    for link in links:
        href = link.get_attribute('href')
        if href and "/team/_/id/" in href:
            try:
                parts = href.split("/")
                team_id = parts[7]
                team_name = parts[8]
                key = (team_id, team_name)
                if key not in seen:
                    seen.add(key)
                    teams.append({"team_id": team_id, "team_name": team_name})
            except:
                continue
    return teams

def get_team_roster(team_id, team_name):
    url = f"https://www.espn.com/mens-college-basketball/team/roster/_/id/{team_id}"
    driver.get(url)
    time.sleep(2)

    players = []
    links = driver.find_elements(By.XPATH, "//a[contains(@href, '/player/_/id/')]")
    for link in links:
        try:
            href = link.get_attribute("href")
            if not href:
                continue
            espn_id = href.split("/")[7]
            player_name = link.text.strip()
            if not player_name or player_name.startswith("(ID:"):
                continue
            players.append({
                "team_id": team_id,
                "team_name": team_name,
                "player_name": player_name,
                "espn_id": espn_id
            })
        except:
            continue
    return players

for team in all_teams:
    print(f"Scraping {team['team_name']}...")
    players = get_team_roster(team["team_id"], team["team_name"])
    
    if players:
        print(f"Found {len(players)} players:")
        for p in players[:3]:
            print(f"  - {p['player_name']} (ID: {p['espn_id']})")
    else:
        print("No players found.")
    
    all_players.extend(players)
    time.sleep(1)

driver.quit()

# Convert to DataFrame and export
df = pd.DataFrame(all_players, columns=["team_id", "team_name", "player_name", "espn_id"])
df.drop_duplicates(subset="espn_id", inplace=True)
print("Rows missing names:", df[df['player_name'] == ""].shape[0])
df.to_csv("espn_ncaa_player_ids.csv", index=False)
print("Saved as espn_ncaa_player_ids.csv")
# -


