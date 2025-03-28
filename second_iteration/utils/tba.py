from mongoengine import *
from dotenv import load_dotenv
import os
import requests
import pandas as pd


connect("reefscape_predictor")
load_dotenv()
tba_api_key = os.getenv("TBA_API_KEY")


class Team(Document):
    event = StringField(required=True)
    team = IntField(required=True)
    epa = FloatField(required=True)
    total_points = FloatField(required=True)
    auto_points = FloatField(required=True)
    teleop_points = FloatField(required=True)
    endgame_points = FloatField(required=True)
    rank = IntField(required=True)
    winrate = FloatField(required=True)
    coral_count = FloatField()
    l4_count = FloatField()
    l3_count = FloatField()
    algae_count = FloatField()
    opr = FloatField()
    ccwm = FloatField()

    def __str__(self):
        return f"Team {self.team} | EPA: {self.epa} | Total: {self.total_points}"


class Event(Document):
    name = StringField(required=True)
    key = StringField(required=True)
    week = IntField(required=True)
    oprs = DictField(required=True)
    coprs = DictField(required=True)

    def __str__(self):
        return f"{self.name} week {self.week} ({self.key})"


class Match(Document):
    key = StringField(required=True)
    red_alliance = ListField(
        ReferenceField(Team), required=True, min_length=3, max_length=3
    )
    blue_alliance = ListField(
        ReferenceField(Team), required=True, min_length=3, max_length=3
    )
    red_score = FloatField(required=True)
    blue_score = FloatField(required=True)

    def __str__(self):
        return (
            f"Match {self.key}: "
            f"Red {[team.team for team in self.red_alliance]} ({self.red_score}) - "
            f"Blue {[team.team for team in self.blue_alliance]} ({self.blue_score})"
        )


class CompleteMatch(Document):
    key = StringField(required=True)
    red3_epa = FloatField(required=True)
    red2_epa = FloatField(required=True)
    red1_epa = FloatField(required=True)
    blue3_epa = FloatField(required=True)
    blue2_epa = FloatField(required=True)
    blue1_epa = FloatField(required=True)
    red3_total_points = FloatField(required=True)
    red2_total_points = FloatField(required=True)
    red1_total_points = FloatField(required=True)
    blue3_total_points = FloatField(required=True)
    blue2_total_points = FloatField(required=True)
    blue1_total_points = FloatField(required=True)
    red3_auto_points = FloatField(required=True)
    red2_auto_points = FloatField(required=True)
    red1_auto_points = FloatField(required=True)
    blue3_auto_points = FloatField(required=True)
    blue2_auto_points = FloatField(required=True)
    blue1_auto_points = FloatField(required=True)
    red3_teleop_points = FloatField(required=True)
    red2_teleop_points = FloatField(required=True)
    red1_teleop_points = FloatField(required=True)
    blue3_teleop_points = FloatField(required=True)
    blue2_teleop_points = FloatField(required=True)
    blue1_teleop_points = FloatField(required=True)
    red3_endgame_points = FloatField(required=True)
    red2_endgame_points = FloatField(required=True)
    red1_endgame_points = FloatField(required=True)
    blue3_endgame_points = FloatField(required=True)
    blue2_endgame_points = FloatField(required=True)
    blue1_endgame_points = FloatField(required=True)
    red3_rank = IntField(required=True)
    red2_rank = IntField(required=True)
    red1_rank = IntField(required=True)
    blue3_rank = IntField(required=True)
    blue2_rank = IntField(required=True)
    blue1_rank = IntField(required=True)
    red3_winrate = FloatField(required=True)
    red2_winrate = FloatField(required=True)
    red1_winrate = FloatField(required=True)
    blue3_winrate = FloatField(required=True)
    blue2_winrate = FloatField(required=True)
    blue1_winrate = FloatField(required=True)
    red3_coral_count = FloatField()
    red2_coral_count = FloatField()
    red1_coral_count = FloatField()
    blue3_coral_count = FloatField()
    blue2_coral_count = FloatField()
    blue1_coral_count = FloatField()
    red3_l4_count = FloatField()
    red2_l4_count = FloatField()
    red1_l4_count = FloatField()
    blue3_l4_count = FloatField()
    blue2_l4_count = FloatField()
    blue1_l4_count = FloatField()
    red3_l3_count = FloatField()
    red2_l3_count = FloatField()
    red1_l3_count = FloatField()
    blue3_l3_count = FloatField()
    blue2_l3_count = FloatField()
    blue1_l3_count = FloatField()
    red3_algae_count = FloatField()
    red2_algae_count = FloatField()
    red1_algae_count = FloatField()
    blue3_algae_count = FloatField()
    blue2_algae_count = FloatField()
    blue1_algae_count = FloatField()
    red3_opr = FloatField()
    red2_opr = FloatField()
    red1_opr = FloatField()
    blue3_opr = FloatField()
    blue2_opr = FloatField()
    blue1_opr = FloatField()
    red3_ccwm = FloatField()
    red2_ccwm = FloatField()
    red1_ccwm = FloatField()
    blue3_ccwm = FloatField()
    blue2_ccwm = FloatField()
    blue1_ccwm = FloatField()
    red_score = FloatField()
    blue_score = FloatField()

    def __str__(self):
        return (
            f"Match {self.key}: Red teams - "
            f"{self.red1_team}, {self.red2_team}, {self.red3_team} ({self.red_score}) vs. "
            f"Blue teams - {self.blue1_team}, {self.blue2_team}, {self.blue3_team} ({self.blue_score})"
        )


def get_all_season_events():
    try:
        req = requests.get(
            "https://www.thebluealliance.com/api/v3/events/2025",
            {"X-TBA-Auth-Key": tba_api_key},
        )
        return req.json()
    except requests.ConnectionError as e:
        print(e)
        return []


def get_oprs(event_key: str):
    try:
        req = requests.get(
            f"https://www.thebluealliance.com/api/v3/event/{event_key}/oprs",
            {"X-TBA-Auth-Key": tba_api_key},
        )
        oprs = req.json()
        req = requests.get(
            f"https://www.thebluealliance.com/api/v3/event/{event_key}/coprs",
            {"X-TBA-Auth-Key": tba_api_key},
        )
        coprs = req.json()
        return oprs, coprs
    except requests.ConnectionError as e:
        print(e)
        return None, None


def save_event(event_key):
    if Event.objects(key=event_key):
        return Event.objects(key=event_key)[0]
    oprs, coprs = get_oprs(event_key)
    res = requests.get(
        f"https://www.thebluealliance.com/api/v3/event/{event_key}",
        {"X-TBA-Auth-Key": tba_api_key},
    )
    response = res.json()
    event_obj = Event(
        week=response["week"],
        name=response["name"],
        key=response["key"],
        oprs=oprs,
        coprs=coprs,
    )
    event_obj.save()
    return event_obj


def get_events_match_keys(event_key: str):
    try:
        req = requests.get(
            f"https://www.thebluealliance.com/api/v3/event/{event_key}/matches/keys",
            {"X-TBA-Auth-Key": tba_api_key},
        )
        return req.json()
    except requests.ConnectionError as e:
        print(f"Error fetching {event_key} keys: {e}")
        return []


def get_team_info(team_number: int, event_key: str):
    mongo_team = Team.objects(team=team_number, event=event_key)
    if mongo_team:
        return mongo_team[0]
    try:
        req = requests.get(f"https://api.statbotics.io/v3/team_year/{team_number}/2025")
        team_dict = req.json()
        if not Event.objects(key=event_key):
            save_event(event_key)
        event_obj = Event.objects(key=event_key)[0]
        ccwm = event_obj.oprs["ccwms"][f"frc{team_number}"]
        opr = event_obj.oprs["oprs"][f"frc{team_number}"]
        l3_count = event_obj.coprs["L3 Coral Count"][f"frc{team_number}"]
        l4_count = event_obj.coprs["L4 Coral Count"][f"frc{team_number}"]
        coral_count = event_obj.coprs["Total Coral Count"][f"frc{team_number}"]
        algae_count = event_obj.coprs["Total Algae Count"][f"frc{team_number}"]
        team_info = {
            "event": event_key,
            "team": team_dict["team"],
            "epa": team_dict["epa"]["norm"],
            "total_points": team_dict["epa"]["breakdown"]["total_points"],
            "auto_points": team_dict["epa"]["breakdown"]["auto_points"],
            "teleop_points": team_dict["epa"]["breakdown"]["teleop_points"],
            "endgame_points": team_dict["epa"]["breakdown"]["endgame_points"],
            "rank": team_dict["epa"]["ranks"]["total"]["rank"],
            "winrate": team_dict["record"]["winrate"],
            "ccwm": ccwm,
            "opr": opr,
            "l3_count": l3_count,
            "l4_count": l4_count,
            "coral_count": coral_count,
            "algae_count": algae_count,
        }
        team_obj = Team.from_json(str(team_info).replace("'", '"'))
        team_obj.save()
        return team_obj
    except Exception as e:
        print(f"Error fetching team {team_number} at {event_key}: {e}")
        return None


def get_match_info(match_key: str):
    mongo_match = Match.objects(key=match_key)
    if mongo_match:
        return mongo_match[0]
    try:
        response = requests.get(
            f"https://www.thebluealliance.com/api/v3/match/{match_key}",
            {"X-TBA-Auth-Key": tba_api_key},
        )
        info = response.json()
        match = Match()
        match.key = match_key
        red_info = info["alliances"]["red"]
        blue_info = info["alliances"]["blue"]

        event_key = match_key[: match_key.index("_")]
        match.red_alliance = [
            get_team_info(int(team[3:]), event_key) for team in red_info["team_keys"]
        ]
        match.blue_alliance = [
            get_team_info(int(team[3:]), event_key) for team in blue_info["team_keys"]
        ]

        match.red_score = red_info.get("score", 0.0)
        match.blue_score = blue_info.get("score", 0.0)
        return match
    except Exception as e:
        print(f"Error fetching match {match_key}: {e}")
        return None


def convert_match_to_complete(match_obj: Match):
    """Convierte un objeto Match en un CompleteMatch y lo guarda en DB."""
    match_data = {
        "key": match_obj.key,
        "red_score": match_obj.red_score,
        "blue_score": match_obj.blue_score,
    }

    for color in ["red", "blue"]:
        alliance = match_obj.red_alliance if color == "red" else match_obj.blue_alliance
        for pos in range(1, 4):
            team_ref = alliance[pos - 1]
            team_data = Team.objects(team=team_ref.team)[0]
            match_data[f"{color}{pos}_team"] = team_ref.team
            match_data[f"{color}{pos}_epa"] = team_data.epa
            match_data[f"{color}{pos}_total_points"] = team_data.total_points
            match_data[f"{color}{pos}_auto_points"] = team_data.auto_points
            match_data[f"{color}{pos}_teleop_points"] = team_data.teleop_points
            match_data[f"{color}{pos}_endgame_points"] = team_data.endgame_points
            match_data[f"{color}{pos}_rank"] = team_data.rank
            match_data[f"{color}{pos}_winrate"] = team_data.winrate
            match_data[f"{color}{pos}_coral_count"] = team_data.coral_count
            match_data[f"{color}{pos}_l4_count"] = team_data.l4_count
            match_data[f"{color}{pos}_l3_count"] = team_data.l3_count
            match_data[f"{color}{pos}_algae_count"] = team_data.algae_count
            match_data[f"{color}{pos}_opr"] = team_data.opr
            match_data[f"{color}{pos}_ccwm"] = team_data.ccwm
    final_match = CompleteMatch(**match_data)
    final_match.save()
    return final_match


def get_match_series(match_key: str) -> pd.Series:
    """Genera una Series con el orden de columnas esperado para un match ya jugado."""
    complete_matches = CompleteMatch.objects(key=match_key)
    if complete_matches:
        complete_match = complete_matches[0]
    else:
        match_obj = get_match_info(match_key)
        if not match_obj:
            raise ValueError(f"Error al obtener match {match_key}")
        complete_match = convert_match_to_complete(match_obj)
        complete_match = CompleteMatch.objects(key=match_key)[0]
    match_dict = complete_match._data.copy()
    if "_id" in match_dict:
        del match_dict["_id"]
    columns_order = [
        "red3_epa",
        "red2_epa",
        "red1_epa",
        "blue3_epa",
        "blue2_epa",
        "blue1_epa",
        "red3_total_points",
        "red2_total_points",
        "red1_total_points",
        "blue3_total_points",
        "blue2_total_points",
        "blue1_total_points",
        "red3_auto_points",
        "red2_auto_points",
        "red1_auto_points",
        "blue3_auto_points",
        "blue2_auto_points",
        "blue1_auto_points",
        "red3_teleop_points",
        "red2_teleop_points",
        "red1_teleop_points",
        "blue3_teleop_points",
        "blue2_teleop_points",
        "blue1_teleop_points",
        "red3_endgame_points",
        "red2_endgame_points",
        "red1_endgame_points",
        "blue3_endgame_points",
        "blue2_endgame_points",
        "blue1_endgame_points",
        "red3_rank",
        "red2_rank",
        "red1_rank",
        "blue3_rank",
        "blue2_rank",
        "blue1_rank",
        "red3_winrate",
        "red2_winrate",
        "red1_winrate",
        "blue3_winrate",
        "blue2_winrate",
        "blue1_winrate",
        "red3_coral_count",
        "red2_coral_count",
        "red1_coral_count",
        "blue3_coral_count",
        "blue2_coral_count",
        "blue1_coral_count",
        "red3_l4_count",
        "red2_l4_count",
        "red1_l4_count",
        "blue3_l4_count",
        "blue2_l4_count",
        "blue1_l4_count",
        "red3_l3_count",
        "red2_l3_count",
        "red1_l3_count",
        "blue3_l3_count",
        "blue2_l3_count",
        "blue1_l3_count",
        "red3_algae_count",
        "red2_algae_count",
        "red1_algae_count",
        "blue3_algae_count",
        "blue2_algae_count",
        "blue1_algae_count",
        "red3_opr",
        "red2_opr",
        "red1_opr",
        "blue3_opr",
        "blue2_opr",
        "blue1_opr",
        "red3_ccwm",
        "red2_ccwm",
        "red1_ccwm",
        "blue3_ccwm",
        "blue2_ccwm",
        "blue1_ccwm",
        "red_score",
        "blue_score",
    ]
    ordered_dict = {col: match_dict[col] for col in columns_order if col in match_dict}
    return pd.Series(ordered_dict)


def get_future_match_series(match_key: str) -> pd.Series:
    """
    Genera una pandas Series para un match que aún no se ha jugado (sin score oficial).
    Se fija en el API de TBA, y en caso de no encontrar score se asigna 0.
    """
    try:
        response = requests.get(
            f"https://www.thebluealliance.com/api/v3/match/{match_key}",
            {"X-TBA-Auth-Key": tba_api_key},
        )
        info = response.json()
        match_data = {"key": match_key}

        match_data["red_score"] = 0.0
        match_data["blue_score"] = 0.0
        event_key = match_key[: match_key.index("_")]
        for color in ["red", "blue"]:
            alliance_info = info["alliances"][color]
            team_keys = alliance_info["team_keys"]
            print("Team keys", team_keys)
            for pos, team_key in enumerate(team_keys, start=1):
                team_number = int(team_key[3:])
                team_obj = get_team_info(team_number, event_key)
                match_data[f"{color}{pos}_team"] = team_number
                match_data[f"{color}{pos}_epa"] = team_obj.epa
                match_data[f"{color}{pos}_total_points"] = team_obj.total_points
                match_data[f"{color}{pos}_auto_points"] = team_obj.auto_points
                match_data[f"{color}{pos}_teleop_points"] = team_obj.teleop_points
                match_data[f"{color}{pos}_endgame_points"] = team_obj.endgame_points
                match_data[f"{color}{pos}_rank"] = team_obj.rank
                match_data[f"{color}{pos}_winrate"] = team_obj.winrate
                match_data[f"{color}{pos}_coral_count"] = team_obj.coral_count
                match_data[f"{color}{pos}_l4_count"] = team_obj.l4_count
                match_data[f"{color}{pos}_l3_count"] = team_obj.l3_count
                match_data[f"{color}{pos}_algae_count"] = team_obj.algae_count
                match_data[f"{color}{pos}_opr"] = team_obj.opr
                match_data[f"{color}{pos}_ccwm"] = team_obj.ccwm
        columns_order = [
            "red3_epa",
            "red2_epa",
            "red1_epa",
            "blue3_epa",
            "blue2_epa",
            "blue1_epa",
            "red3_total_points",
            "red2_total_points",
            "red1_total_points",
            "blue3_total_points",
            "blue2_total_points",
            "blue1_total_points",
            "red3_auto_points",
            "red2_auto_points",
            "red1_auto_points",
            "blue3_auto_points",
            "blue2_auto_points",
            "blue1_auto_points",
            "red3_teleop_points",
            "red2_teleop_points",
            "red1_teleop_points",
            "blue3_teleop_points",
            "blue2_teleop_points",
            "blue1_teleop_points",
            "red3_endgame_points",
            "red2_endgame_points",
            "red1_endgame_points",
            "blue3_endgame_points",
            "blue2_endgame_points",
            "blue1_endgame_points",
            "red3_rank",
            "red2_rank",
            "red1_rank",
            "blue3_rank",
            "blue2_rank",
            "blue1_rank",
            "red3_winrate",
            "red2_winrate",
            "red1_winrate",
            "blue3_winrate",
            "blue2_winrate",
            "blue1_winrate",
            "red3_coral_count",
            "red2_coral_count",
            "red1_coral_count",
            "blue3_coral_count",
            "blue2_coral_count",
            "blue1_coral_count",
            "red3_l4_count",
            "red2_l4_count",
            "red1_l4_count",
            "blue3_l4_count",
            "blue2_l4_count",
            "blue1_l4_count",
            "red3_l3_count",
            "red2_l3_count",
            "red1_l3_count",
            "blue3_l3_count",
            "blue2_l3_count",
            "blue1_l3_count",
            "red3_algae_count",
            "red2_algae_count",
            "red1_algae_count",
            "blue3_algae_count",
            "blue2_algae_count",
            "blue1_algae_count",
            "red3_opr",
            "red2_opr",
            "red1_opr",
            "blue3_opr",
            "blue2_opr",
            "blue1_opr",
            "red3_ccwm",
            "red2_ccwm",
            "red1_ccwm",
            "blue3_ccwm",
            "blue2_ccwm",
            "blue1_ccwm",
            "red_score",
            "blue_score",
        ]
        ordered_dict = {
            col: match_data[col] for col in columns_order if col in match_data
        }
        return pd.Series(ordered_dict)
    except Exception as e:
        print(f"Error in get_future_match_series: {e}")
        return None
