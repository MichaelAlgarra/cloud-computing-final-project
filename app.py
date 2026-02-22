"""
app.py
Main entrypoint for the MLB Player Analyzer Flask APp.

This is an application that utilizes Google Gemini in Google Cloud as well as the pybaseball package.
This will give a user a selection from a team and year, allow them to select a player, and then receive a letter grade and summary for their success.
Referenced from : https://github.com/jldbc/pybaseball

By Michael Algarra for DATS 5750
"""
from flask import Flask, render_template, request, jsonify
import pybaseball
import pandas as pd
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

pybaseball.cache.enable()

app = Flask(__name__)

GCP_PROJECT  = os.environ.get("GCP_PROJECT")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

_gemini = None
def get_gemini():
    global _gemini
    if _gemini is None:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        _gemini = GenerativeModel("gemini-2.0-flash-001") # Utilizing Gemini 2.0
    return _gemini


MLB_TEAMS = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}

AVAILABLE_YEARS = list(range(2025, 2014, -1))


@app.route("/")
def index():
    teams_sorted = dict(sorted(MLB_TEAMS.items(), key=lambda x: x[1]))
    return render_template("index.html", teams=teams_sorted, years=AVAILABLE_YEARS)


@app.route("/api/players", methods=["GET"])
def get_players():
    team = request.args.get("team")
    year = request.args.get("year", type=int)

    if not team or not year:
        return jsonify({"error": "Team and year are required"}), 400

    try:
        batters_df  = pybaseball.batting_stats(year, year, qual=1)
        pitchers_df = pybaseball.pitching_stats(year, year, qual=1)

        team_batters  = batters_df[batters_df["Team"] == team].copy()
        team_pitchers = pitchers_df[pitchers_df["Team"] == team].copy()

        batters_list = []
        for _, row in team_batters.iterrows():
            batters_list.append({
                "name": row["Name"],
                "type": "Batter",
                "games": int(row.get("G", 0)),
                "avg":   round(float(row.get("AVG", 0)), 3),
                "hr":    int(row.get("HR", 0)),
                "rbi":   int(row.get("RBI", 0)),
                "ops":   round(float(row.get("OPS", 0)), 3),
            })

        pitchers_list = []
        for _, row in team_pitchers.iterrows():
            pitchers_list.append({
                "name":       row["Name"],
                "type":       "Pitcher",
                "games":      int(row.get("G", 0)),
                "era":        round(float(row.get("ERA", 0)), 2),
                "wins":       int(row.get("W", 0)),
                "strikeouts": int(row.get("SO", 0)),
                "whip":       round(float(row.get("WHIP", 0)), 2),
            })

        batters_list.sort(key=lambda x: x["games"], reverse=True)
        pitchers_list.sort(key=lambda x: x["games"], reverse=True)

        return jsonify({
            "team":     MLB_TEAMS.get(team, team),
            "year":     year,
            "batters":  batters_list,
            "pitchers": pitchers_list,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/player-stats", methods=["GET"])
def get_player_stats():
    player_name = request.args.get("name")
    year        = request.args.get("year", type=int)
    player_type = request.args.get("type", "Batter")

    if not player_name or not year:
        return jsonify({"error": "Player name and year are required"}), 400

    try:
        if player_type == "Batter":
            df         = pybaseball.batting_stats(year, year, qual=1)
            player_row = df[df["Name"] == player_name]

            if player_row.empty:
                return jsonify({"error": "Player not found"}), 404

            row   = player_row.iloc[0]
            stats = {
                "name":         player_name,
                "year":         year,
                "type":         "Batter",
                "team":         row.get("Team", "N/A"),
                "games":        int(row.get("G", 0)),
                "at_bats":      int(row.get("AB", 0)),
                "hits":         int(row.get("H", 0)),
                "doubles":      int(row.get("2B", 0)),
                "triples":      int(row.get("3B", 0)),
                "home_runs":    int(row.get("HR", 0)),
                "rbi":          int(row.get("RBI", 0)),
                "stolen_bases": int(row.get("SB", 0)),
                "walks":        int(row.get("BB", 0)),
                "strikeouts":   int(row.get("SO", 0)),
                "avg":          round(float(row.get("AVG", 0)), 3),
                "obp":          round(float(row.get("OBP", 0)), 3),
                "slg":          round(float(row.get("SLG", 0)), 3),
                "ops":          round(float(row.get("OPS", 0)), 3),
                "war":          round(float(row.get("WAR", 0)), 1),
                "wrc_plus":     int(row.get("wRC+", 0)) if "wRC+" in row else "N/A",
            }
        else:
            df         = pybaseball.pitching_stats(year, year, qual=1)
            player_row = df[df["Name"] == player_name]

            if player_row.empty:
                return jsonify({"error": "Player not found"}), 404

            row   = player_row.iloc[0]
            stats = {
                "name":            player_name,
                "year":            year,
                "type":            "Pitcher",
                "team":            row.get("Team", "N/A"),
                "games":           int(row.get("G", 0)),
                "games_started":   int(row.get("GS", 0)),
                "wins":            int(row.get("W", 0)),
                "losses":          int(row.get("L", 0)),
                "saves":           int(row.get("SV", 0)),
                "innings_pitched": round(float(row.get("IP", 0)), 1),
                "strikeouts":      int(row.get("SO", 0)),
                "walks":           int(row.get("BB", 0)),
                "era":             round(float(row.get("ERA", 0)), 2),
                "whip":            round(float(row.get("WHIP", 0)), 2),
                "k9":              round(float(row.get("K/9",  0)), 1) if "K/9"  in row else "N/A",
                "bb9":             round(float(row.get("BB/9", 0)), 1) if "BB/9" in row else "N/A",
                "fip":             round(float(row.get("FIP",  0)), 2) if "FIP"  in row else "N/A",
                "war":             round(float(row.get("WAR",  0)), 1),
            }

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_player():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name        = data.get("name")
    year        = data.get("year")
    player_type = data.get("type")

    if not name or not year or not player_type:
        return jsonify({"error": "name, year, and type are required"}), 400

    try:
        if player_type == "Batter":
            stat_lines = [
                f"Games: {data.get('games')}",
                f"At-Bats: {data.get('at_bats')}",
                f"Hits: {data.get('hits')}",
                f"Batting Average (AVG): {data.get('avg')}",
                f"On-Base Percentage (OBP): {data.get('obp')}",
                f"Slugging Percentage (SLG): {data.get('slg')}",
                f"OPS: {data.get('ops')}",
                f"Home Runs (HR): {data.get('home_runs')}",
                f"RBI: {data.get('rbi')}",
                f"Stolen Bases (SB): {data.get('stolen_bases')}",
                f"Walks (BB): {data.get('walks')}",
                f"Strikeouts (SO): {data.get('strikeouts')}",
                f"Doubles (2B): {data.get('doubles')}",
                f"Triples (3B): {data.get('triples')}",
                f"WAR: {data.get('war')}",
                f"wRC+: {data.get('wrc_plus')}",
            ]
            role_context     = "position player (batter)"
            grade_benchmarks = (
                "Use these grade benchmarks for a position player:\n"
                "  A+ = MVP-caliber (WAR 7+, OPS 1.000+)\n"
                "  A  = All-Star level (WAR 5-7, OPS .900-.999)\n"
                "  B  = Above average starter (WAR 3-5, OPS .800-.899)\n"
                "  C  = Average MLB starter (WAR 1-3, OPS .700-.799)\n"
                "  D  = Below average / fringe starter (WAR 0-1, OPS .600-.699)\n"
                "  F  = Replacement level or worse (WAR < 0, OPS below .600)"
            )
        else:
            stat_lines = [
                f"Games: {data.get('games')}",
                f"Games Started (GS): {data.get('games_started')}",
                f"Wins: {data.get('wins')}",
                f"Losses: {data.get('losses')}",
                f"Saves: {data.get('saves')}",
                f"Innings Pitched (IP): {data.get('innings_pitched')}",
                f"ERA: {data.get('era')}",
                f"WHIP: {data.get('whip')}",
                f"Strikeouts (K): {data.get('strikeouts')}",
                f"Walks (BB): {data.get('walks')}",
                f"K/9: {data.get('k9')}",
                f"BB/9: {data.get('bb9')}",
                f"FIP: {data.get('fip')}",
                f"WAR: {data.get('war')}",
            ]
            role_context     = "pitcher"
            grade_benchmarks = (
                "Use these grade benchmarks for a pitcher:\n"
                "  A+ = Cy Young-caliber (WAR 7+, ERA sub-2.50)\n"
                "  A  = Top-of-rotation / elite closer (WAR 4-7, ERA 2.50-3.25)\n"
                "  B  = Solid starter or high-leverage reliever (WAR 2-4, ERA 3.25-3.75)\n"
                "  C  = League-average (WAR 1-2, ERA 3.75-4.50)\n"
                "  D  = Below average (WAR 0-1, ERA 4.50-5.50)\n"
                "  F  = Replacement level or worse (WAR < 0, ERA 5.50+)"
            )

        stats_block = "\n".join(stat_lines)

        prompt = f"""You are an expert MLB baseball analyst with deep knowledge of sabermetrics and historical player performance.

Analyze the following {year} season statistics for {name}, a {role_context}.

{grade_benchmarks}

{year} Season Stats:
{stats_block}

Your response MUST follow this exact format with these two sections:

SUMMARY:
Write 2-3 paragraphs evaluating the player's season. Discuss their strengths, weaknesses, and how their numbers compare to league averages and position-adjusted expectations. Reference specific stats to support your analysis. Keep the tone professional but engaging, like a baseball analyst writing for a knowledgeable audience.

GRADE:
Assign a single letter grade (A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, or F) based on the benchmarks above. Then write one sentence explaining the grade.

Do not include any other sections or commentary outside of SUMMARY and GRADE."""

        response     = get_gemini().generate_content(prompt)
        raw          = response.text.strip()
        summary      = ""
        grade_text   = ""
        grade_letter = ""

        if "SUMMARY:" in raw and "GRADE:" in raw:
            summary_start = raw.index("SUMMARY:") + len("SUMMARY:")
            grade_start   = raw.index("GRADE:")
            summary       = raw[summary_start:grade_start].strip()
            grade_block   = raw[grade_start + len("GRADE:"):].strip()

            first_token  = grade_block.split()[0] if grade_block else ""
            valid_grades = {"A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","F"}
            if first_token in valid_grades:
                grade_letter = first_token
                grade_text   = grade_block
            else:
                for g in ["A+","A-","B+","B-","C+","C-","D+","D-","A","B","C","D","F"]:
                    if g in grade_block:
                        grade_letter = g
                        grade_text   = grade_block
                        break
        else:
            summary    = raw
            grade_text = "Grade not detected."

        return jsonify({
            "summary":    summary,
            "grade":      grade_letter,
            "grade_text": grade_text,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
