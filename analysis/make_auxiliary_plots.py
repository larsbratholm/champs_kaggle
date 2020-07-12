"""
Code to plot the progression of the public leaderboard and how the number of teams relates to previous competitions.
Partly based on https://www.kaggle.com/robikscube/the-race-to-predict-molecular-properties
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
import pandas as pd
import itertools
#import time
import datetime
import scipy
import os

#Set plotting theme
sns.set(font_scale=2.,rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
rc('text', usetex=False)
# Set "b", "g", "r" to default seaborn colors
sns.set_color_codes("deep")

def read_and_process_data(filename):
    df = pd.read_csv(filename)
    df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
    # Needed to plot submissions per day
    df_unfiltered = df.set_index(['SubmissionDate'])

    df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
    df.columns = [name for name in df.columns]

    df_filtered = df.ffill()
    return df_unfiltered, df_filtered

def plot_progress_all_teams(df, public_kernels, filename, best_score=False):
    """
    Shows progress of all teams.
    best_score enables whether the best score over time is tracked,
    or just submissions. If set to True, the plot will take a long
    time to finish rendering.
    """
    plt.figure(figsize=(16,9))
    best = df.min(axis=1)

    # Force red color to all lines
    palette = itertools.cycle(sns.xkcd_palette(["pale red"]))
    n = df.shape[1]
    step = 1
    #t0 = time.time()
    # Hack to get "All teams" legend
    ax = sns.lineplot(x = [datetime.date(year=2019,month=7,day=1)]*2, y=[100,100], alpha=1,
            color="r", dashes=False, label="All teams")
    for i in range(n // step +min(1, n % step)):
        #t1 = time.time()
        #if i % 100 == 0:
        #    print("%.2f percent" % (100*step*i/n), "%.2f seconds" % (t1 - t0))
        if best_score:
            sns.lineplot(data=df.iloc[:,step*i:step*(i+1)], alpha=0.05, palette=palette,
                    ax=ax, legend=False, dashes=False)
        else:
            sns.lineplot(data=df.iloc[:,step*i:step*(i+1)].dropna().drop_duplicates(keep="first"), alpha=0.05, palette=palette,
                    ax=ax, legend=False, dashes=False)

    sns.lineplot(ax=ax, data=best, color="k",
            label="Leader", alpha=0.8)

    sns.lineplot(ax=ax, data=public_kernels, color="b",
            alpha=0.8)
    format_and_save(ax, filename)

def plot_progress_select_teams(df, filename):
    plt.figure(figsize=(16,9))
    team_ordered = df.loc[df.index.max()] \
            .sort_values(ascending=True).index.tolist()
    select_teams = team_ordered[:5] + [team_ordered[11]]

    df_subset = df[select_teams]
    df_subset.columns = ["#1", "#2", "#3", "#4", "#5", "#12"]

    best = df.min(axis=1)

    ax = sns.lineplot(data=best, color="k",
            label="Leader", alpha=0.8)
    palette = itertools.cycle(sns.color_palette("deep",3))
    sns.lineplot(ax=ax, data=df_subset, alpha=0.8,
            palette=palette,
            dashes=[(1,0)]*3+[(1,1)]*3)

    format_and_save(ax, filename)

def format_and_save(ax, filename, clear=True, ylabel="Score", ylim=(-3.6,1)):
    ax.set(xlabel='Submission date', ylabel=ylabel, ylim=ylim,
            xlim=(datetime.date(2019,5,29),datetime.date(2019,9,2)))
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m/%y"))
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.savefig(filename, pad_inches=0.0, bbox_inches="tight", dpi=300)
    if clear:
        plt.clf()

def read_public_kernels(filename):
    df = pd.read_csv(filename)
    # Add 18 hours, since the dates are UK time at midnight
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y") \
            + datetime.timedelta(hours=18)
    # Keep only best scores from a given date
    df.sort_values("date", inplace=True)
    for index, item in df.iterrows():
        score = item["public score"]
        date = item.date
        mask = (df.date <= date)
        min_score_seen = df[mask]["public score"].min()
        if score > min_score_seen:
            df.drop(index, inplace=True)

    df["TeamName"] = "Best public score"
    df = df.sort_values("public score").drop_duplicates("date", keep="first")
    df = df.set_index(['TeamName','date'])['public score'].unstack(-1).T
    df.columns = [name for name in df.columns]

    df_filtered = df.ffill().min(1)
    return df

def plot_number_of_teams(df, filename):
    plt.figure(figsize=(16,5))
    df.count(axis=1)
    ax = sns.lineplot(data=df.count(axis=1), label="Number of teams")
    format_and_save(ax, filename, ylabel="Count", ylim=None)

def plot_days_between_submissions(df, filename, truncate):
    plt.figure(figsize=(16,9))
    #ax = sns.lineplot(data=df.T.iloc[0].T.dropna().drop_duplicates(),
    #        legend=False, dashes=False)
    #for i in range(1,10):
    #    sns.lineplot(data=df.T.iloc[i].T.dropna().drop_duplicates(), 
    #        legend=False, dashes=False, ax=ax)
    #plt.show()
    days = np.empty(2723)
    for i in range(2723):
        team_df = df.T.iloc[i].T.dropna().drop_duplicates()
        delta = team_df.axes[0].max() - team_df.axes[0].min()
        n_days = delta.total_seconds() / 3600 / 24
        if truncate:
            n_days = min(n_days, 21)
        days[i] = n_days


    # - 0.5 to get center aligned bins
    plt.hist(days - 0.5, bins=int(max(days)+0.5))
    plt.gca().set(xlabel='Days between first and last submission', ylabel="Number of teams")
    #plt.yscale('log')
    plt.savefig(filename)
    plt.clf()

def plot_submissions_per_day(df, filename):
    plt.figure(figsize=(16,9))

    # Remove dummy entries
    df = df[2736:]

    # Get submissions per day (ignore first day)
    df = df.resample('D').apply({'Score':'count'})[1:]

    sns.lineplot(df.index, df.Score)
    sns.scatterplot(df.index, df.Score, s=60)
    format_and_save(plt.gca(), filename, ylabel="Count", ylim=None)

def plot_exponential_fits(df, filename):
    def opt(params, t, y_true, fun):
        y = fun(params, t)
        return 1/y.size * sum((y-y_true)**2)

    def single(params, t):
        """
        Single exponential
        """
        A, a, C = params
        return A*np.exp(-a*t) + C

    def double(params, t):
        """
        Double exponential
        """
        A, a, C, B, b = params
        return A*np.exp(-t/a) + B*np.exp(-t/b) + C

    def convert_df(df):
        """
        Converts the dates and values of the dataframe
        to something more easily plotted
        """
        dates = df.axes[0]
        x = (dates - datetime.datetime.utcfromtimestamp(0)).total_seconds().values
        x -= x.min()
        x /= (3600*24)
        y = df.values
        return x, y, dates

    best = df.min(1)
    # Only keep changes in leaderboard
    best_unique = best.drop_duplicates(keep="first")

    x, y, dates = convert_df(best)
    x_unique, y_unique, dates_unique = convert_df(best_unique)

    params0 = [3, 1, -3.5, 2, 50]
    bounds = [(0,None), (0,None), (None,None), (0,None), (0,None)]
    params = scipy.optimize.minimize(opt, params0, args=(x_unique, y_unique, double),
            options={"maxiter":10000}, bounds=bounds, tol=1e-6, method='slsqp')
    assert params.success, params
    print("Fitted parameters", params.x)
    # Estimated from fit_leastsq of https://stackoverflow.com/a/21844726/2653663
    print("Errors: [0.178 ,  0.178,  0.288,  0.215, 16.6]")

    plt.figure(figsize=(16,9))
    sns.lineplot(x=dates, y=y, color="k", label="Leader")
    sns.lineplot(x=[dates.min(), dates.max()], y=[-3.453]*2, label="Best ME")
    sns.lineplot(x=dates, y=single((params.x[[0,1,2]]), x), label="Component 1")
    sns.lineplot(x=dates, y=single((params.x[[3,4,2]]), x), label="Component 2")
    sns.lineplot(x=dates, y=double(params.x, x), label="Fit")
    # Reorder legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    order = [0,1,4,2,3]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # Set ylim and ax labels
    format_and_save(ax, filename)

def parse_kaggle_competitions(filename):
    with open(filename) as f:
        lines = f.readlines()
        c = 0
        titles = []
        abstracts = []
        types = []
        months = []
        keywords = []
        prizes = []
        n_teams = []
        limited = []
        for i, line in enumerate(lines):
            c += 1
            if c == 1:
                title = line.strip()
            elif c == 4:
                abstract = line.strip()
            elif c == 5:
                type_and_date = line.strip()
                if "Analytics" in type_and_date:
                    type_ = "Analytics"
                    date_raw = type_and_date[9:]
                elif "FeaturedCode" in type_and_date:
                    type_ = "FeaturedCode"
                    date_raw = type_and_date.split(".")[-1]
                elif "Featured" in type_and_date:
                    type_ = "Featured"
                    date_raw = type_and_date[8:]
                elif "ResearchCode" in type_and_date:
                    type_ = "ResearchCode"
                    date_raw = type_and_date.split(".")[-1]
                elif "Research" in type_and_date:
                    type_ = "Research"
                    date_raw = type_and_date[8:]
                elif "PlaygroundCode" in type_and_date:
                    type_ = "PlaygroundCode"
                    date_raw = type_and_date.split(".")[-1]
                elif "Playground" in type_and_date:
                    type_ = "Playground"
                    date_raw = type_and_date[10:]
                elif "Recruitment" in type_and_date:
                    type_ = "Recruitment"
                    date_raw = type_and_date[11:]
                elif "Getting Started" in type_and_date:
                    type_ = "Getting Started"
                    date_raw = type_and_date[15:]
                elif "Masters" in type_and_date:
                    type_ = "Masters"
                    date_raw = type_and_date[7:]
                else:
                    print("Unknown type and date:", type_and_date)
                    quit()
                if "Limited" in date_raw:
                    lim = True
                    date_raw = date_raw.replace("Limited","")
                else:
                    lim = False

                tokens = date_raw.split()
                if tokens[0] == "a":
                    tokens[0] = 1
                number = int(tokens[0])
                if "year" in date_raw:
                    n_months = number * 12
                elif "month" in date_raw:
                    n_months = number
                elif "days" in date_raw:
                    n_months = number * 12 / 365.2425
                else:
                    print("Unknown date", date_raw)
                    quit()
            elif c == 6:
                if len(line.strip().split()) == 0:
                    c += 1
                keyword = line.strip().split(",")
            elif c == 8:
                prize = line.strip()
                if "team" not in lines[i+1]:
                    c = 0
                    teams = 0
            elif c == 9:
                teams = int(line.split()[0].replace(",",""))
                c = 0

            if c == 0:
                titles.append(title)
                abstracts.append(abstract)
                types.append(type_)
                months.append(n_months)
                keywords.append(keyword)
                prizes.append(prize)
                n_teams.append(teams)
                limited.append(lim)
    return np.asarray(titles), np.asarray(abstracts), np.asarray(types), np.asarray(months), \
            np.asarray(keywords), np.asarray(prizes), np.asarray(n_teams), np.asarray(limited)

def preprocess_kaggle_competitions(titles, abstracts, types, months, keywords, prizes, n_teams, limited):
    # Only look at competition with monetary prizes
    prize_dollars = []
    prize_dollars_idx = []
    for i, prize in enumerate(prizes):
        if "," in prize:
            prize_dollars_idx.append(i)
            if "$" in prize:
                dollars = int(prize[1:].replace(",", ""))
            else:
                # euro for one comp. 1.11 correspondence
                dollars = int(prize[1:].replace(",", "")) * 1.11
            prize_dollars.append(dollars)
        else:
            prize_dollars.append(0)
    prize_dollars = np.asarray(prize_dollars)

    # Only look at competitions open to everyone
    limited_idx = np.where(~limited)[0]

    # Only look at competitions where the number of teams are listed
    team_idx = np.where(n_teams > 0)[0]

    # Ignore recruitment and Code-type competitions
    type_idx = [i for i, x in enumerate(types) if x not in ['FeaturedCode', 'ResearchCode', 'Recruitment']]

    # Only look at last 5 years
    month_idx = np.where(months <= 5*12)[0]

    idx = np.asarray(list(set(prize_dollars_idx) & set(limited_idx) & set(team_idx) & set(type_idx) & set(month_idx)))

    champs_idx = [i for i,x in enumerate(titles) if "predicting molecular" in x.lower()]

    return champs_idx, idx, prize_dollars

def make_teams_vs_prize_plot(script_dir):
    # Parse the website dump
    titles, abstracts, types, months, keywords, prizes, n_teams, limited = \
            parse_kaggle_competitions(f'{script_dir}/data/kaggle_competitions.txt')
    # Preprocess the data (convert currency, retrieve valid entries)
    champs_idx, idx, prize_dollars = preprocess_kaggle_competitions(titles, abstracts, \
            types, months, keywords, prizes, n_teams, limited)
    # plot
    plot_kaggle_competitions(prize_dollars, n_teams, idx, champs_idx, f'{script_dir}/output/prize_vs_teams.png')

def plot_kaggle_competitions(prize_dollars, n_teams, idx, champs_idx, filename):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.scatter((prize_dollars[idx]), (n_teams[idx]), label="All")
    ax.scatter((prize_dollars[champs_idx]), (n_teams[champs_idx]), label="CHAMPS")
    # Log-log scale
    ax.loglog()
    # Remove offset (show full value, not factor of 1e3 etc.)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(useOffset=False, style="plain")
    # Set axis ticks
    ax.set_xticks([1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    ax.set_yticks([100, 300, 1000, 3000, 10000])
    # Make comma separator of thousands
    ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))

    # Rotate xticks
    plt.xticks(rotation=45, horizontalalignment='right')
    # Set label text
    ax.set(xlabel='Prize pool ($)', ylabel="Number of teams")
    # Make legend
    plt.legend()
    # Save figure
    plt.savefig(filename, pad_inches=0.0, bbox_inches="tight", dpi=300)
    plt.clf()

def make_progression_plots(script_dir):
    output_dir = f'{script_dir}/output/'
    data_dir = f'{script_dir}/data/'
    public_kernels = read_public_kernels(data_dir + 'public_kernels.csv')
    df_unfiltered, df = read_and_process_data(data_dir + 'champs-scalar-coupling-publicleaderboard_with_dummies.csv')
    plot_days_between_submissions(df, output_dir + 'days_between_submissions.png', False)
    plot_days_between_submissions(df, output_dir + 'days_between_submissions_truncated.png', True)
    plot_exponential_fits(df, output_dir + 'exponential_fits.png')
    plot_number_of_teams(df, output_dir + 'number_of_teams.png')
    plot_progress_select_teams(df, output_dir + "progress_select_teams.png")
    plot_progress_all_teams(df, public_kernels, output_dir + "progress_all_teams.png")
    plot_submissions_per_day(df_unfiltered, output_dir + "submissions_per_day.png")

if __name__ == "__main__":
    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # Make plot that shows number of teams vs prize pool
    make_teams_vs_prize_plot(script_dir)
    # Make plots related to progression of public leaderboard
    make_progression_plots(script_dir)
