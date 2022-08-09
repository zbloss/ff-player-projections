import os
import click
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

def convert_height_to_inches(height_string: str) -> int:
    """
    Converts the height string (Feet - Inches) into inches.
    
    Arguments:
        height_string (str): Player height as a string in the format of height-inches.
        
    Returns:
        int: Player height in inches.
    
    """
        
    height_string_split = height_string.split('-')
    
    feet, inches = height_string_split[0], height_string_split[1]
    feet, inches = int(feet), int(inches)
    height = feet * 12 + inches
    return height

def pull_data_from_nfl_data_py(start_year: int, end_year: int) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    helper function to pull data.

    Arguments:
        start_year (int): The first year to pull data from.
        end_year (int): The last year to pull data from.

    Returns:
        list: List of pandas Dataframes containing NFL data.

    """
    assert end_year > start_year, f'End year must be greater than start year\nstart_year={start_year}\nend_year=={end_year}'
    years_to_analyze = range(start_year, end_year)

    weekly_data = nfl.import_weekly_data(years=years_to_analyze)
    roster_data = nfl.import_rosters(years=years_to_analyze)
    snap = nfl.import_snap_counts(years=years_to_analyze)
    team_info = nfl.import_team_desc()
    return [weekly_data, roster_data, snap, team_info]

@click.command()
@click.option('--start_year', default=2013, help='First year to pull metrics from')
@click.option('--end_year', default=2021, help='Last year to pull metrics from')
@click.option('--data_filepath', default='data/data.csv', help='Filepath to save the processed output to.')
def process_data(start_year: int, end_year: int, data_filepath: str):
    """
    Downloads, processes, and saves NFL data.

    Arguments:
        start_year (int): The first year to pull data from.
        end_year (int): The last year to pull data from.

    Returns:
        None: None.

    """

    click.echo('Checking filepaths...')
    if os.path.isdir(data_filepath):
        filename = 'data.csv'
    else:
        filepath, filename = os.path.split(data_filepath)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

    click.echo('Downloading raw data...')
    weekly_data, roster_data, snap, team_info = pull_data_from_nfl_data_py(
        start_year,
        end_year
    )

    click.echo('Processing raw data...')
    summed = weekly_data.groupby(['player_id', 'season', 'week']).sum().reset_index()
    data = pd.merge(
        left=summed, 
        right=roster_data, 
        left_on='player_id', 
        right_on='player_id', 
        how='inner', 
        suffixes=('', '_DROP')
    ).filter(regex='^(?!.*_DROP)')

    columns_to_drop = [
        'sacks',
        'sack_yards',
        'sack_fumbles',
        'espn_id',
        'sportradar_id',
        'yahoo_id',
        'rotowire_id',
        'pff_id',
        'fantasy_data_id',
        'sleeper_id',
        'esb_id',
        'gsis_it_id',
        'smart_id',
        'ngs_position',
        'entry_year',
        'player_name',
        'birth_date',
        'jersey_number',
        'special_teams_tds',
        'college',
        'depth_chart_position',
    ]                  
    data.drop(columns_to_drop, axis=1, inplace=True)
    data = data[
        data['position'].isin(['WR', 'RB', 'TE', 'QB'])
    ]
    data = data[~data['rookie_year'].isna()]
    data['rookie_year'] = data['rookie_year'].astype(int)
    data['years_exp'] = (data['season'].astype(int) - data['rookie_year'].astype(int))
    data['height'] = data['height'].apply(lambda height: convert_height_to_inches(height))
    data.drop('weight', axis=1, inplace=True)
    data['first_name'] = data['first_name'].str.lower()
    data['last_name'] = data['last_name'].str.lower()

    data.loc[data['fantasy_points'] > 0, 'status'] = 'Active'
    data.loc[data['fantasy_points_ppr'] > 0, 'status'] = 'Active'

    rows_to_drop = data.loc[
        (data['fantasy_points'] <= 0) & (data['status'] != 'Active'),
        'status'
    ].index
    data.drop(rows_to_drop, axis=0, inplace=True)
    data.drop('status', axis=1, inplace=True)

    snap = snap[snap['position'].isin(data['position'].unique())]
    data = pd.merge(
        left=data,
        right=snap,
        left_on=['pfr_id', 'season', 'week'],
        right_on=['pfr_player_id', 'season', 'week'],
        how='inner', 
        suffixes=('', '_DROP')
    ).filter(regex='^(?!.*_DROP)')
    data = data[data['game_type'] == 'REG']

    columns_to_drop = [
        'pfr_id',
        'game_id',
        'pfr_game_id',
        'game_type',
        'player',
        'pfr_player_id',
        'defense_snaps',
        'defense_pct',
        'st_snaps',
        'st_pct',
        
    ]
    data.drop(
        columns_to_drop,
        axis=1,
        inplace=True
    )

    data = pd.merge(
        left=data,
        right=team_info[['team_abbr', 'team_conf', 'team_division']],
        left_on='team',
        right_on='team_abbr',
        how='inner',
    )
    data = pd.merge(
        left=data,
        right=team_info[['team_abbr', 'team_conf', 'team_division']],
        left_on='opponent',
        right_on='team_abbr',
        how='inner',
    ).rename(
        columns={
            'team_abbr_y': 'opponent_abbr', 
            'team_conf_y': 'opponent_conf', 
            'team_division_y': 'opponent_division',
            'team_abbr_x': 'team_abbr',
            'team_conf_x': 'team_conf',
            'team_division_x': 'team_division'
        }
    )
    data.drop(
        ['team_abbr', 'opponent_abbr'],
        axis=1, 
        inplace=True
    )

    data['division_matchup'] = np.where(
        (data['team_division'] == data['opponent_division']),
        1, 
        0
    )

    data['conference_matchup'] = np.where(
        (data['team_conf'] == data['opponent_conf']),
        1, 
        0
    )

    data = pd.merge(
        left=data,
        right=pd.get_dummies(data['team_conf'], prefix='team'),
        left_index=True,
        right_index=True,
        how='inner'
    )

    data = pd.merge(
        left=data,
        right=pd.get_dummies(data['opponent_conf'], prefix='opponent'),
        left_index=True,
        right_index=True,
        how='inner'
    )

    data = pd.merge(
        left=data,
        right=pd.get_dummies(data['team_division'], prefix='team'),
        left_index=True,
        right_index=True,
        how='inner'
    )

    data = pd.merge(
        left=data,
        right=pd.get_dummies(data['opponent_division'], prefix='opponent'),
        left_index=True,
        right_index=True,
        how='inner'
    )

    data = pd.merge(
        left=data,
        right=pd.get_dummies(data['position']),
        left_index=True,
        right_index=True,
        how='inner'
    )

    columns_to_drop = [
        'team_conf',
        'team_division',
        'opponent_conf',
        'opponent_division',
        'position',
        'headshot_url',
        
    ]
    data.drop(
        columns_to_drop,
        axis=1,
        inplace=True
    )
    data.insert(0, 'player_id', data.pop('player_id'))
    data.insert(1, 'first_name', data.pop('first_name'))
    data.insert(2, 'last_name', data.pop('last_name'))
    data.insert(3, 'team', data.pop('team'))
    data.insert(4, 'opponent', data.pop('opponent'))

    data.to_csv(
        os.path.join(filepath, filename), 
        index=None
    )
    click.echo('Done!')

if __name__ == '__main__':
    process_data()