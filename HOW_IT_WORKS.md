# How the March Madness Prediction Pipeline Works
### Plain English Edition

---

## The Big Picture

We're trying to answer one question: **"If Team A plays Team B, what's the probability Team A wins?"**

We do this by looking at everything we know about both teams heading into the tournament — how good they are, how they've been playing lately, what the computers think of them, what seed they got — and boiling all of it down into a single number between 0 and 1.

The machine learning model doesn't watch film. It doesn't read injury reports. It just looks at numbers and finds patterns in 40 years of historical tournament data: *"Teams with these kinds of numbers tend to win. Teams with those kinds of numbers tend to lose."*

---

## Step 1: The Raw Data

We have 35 CSV files from Kaggle covering every NCAA game since 1985 (men) and 1998 (women). The key ones are:

- **Box scores** — every field goal, rebound, turnover, foul from every game
- **Tournament seeds** — who the committee seeded 1 through 16
- **Massey Ordinals** — computer rankings from 196 different rating systems
- **Game locations** — home, away, or neutral court

Everything we compute flows from these raw files.

---

## Step 2: The Features

A "feature" is just a number that describes something about a team. We build ~38 features per team, then for each matchup we compute the **difference** between the two teams on each feature.

So instead of "Team A has an efficiency rating of 112," the model sees "Team A is 15 points better than Team B on efficiency." Positive = Team A (lower ID) is better. Negative = Team B is better.

Here's what each feature actually means:

---

### The Basics

**Win Percentage**
Exactly what it sounds like — what fraction of regular season games did the team win? A team that went 28-3 has a higher win% than a team that went 18-13.

**Points Per Game / Points Allowed Per Game**
How many points does the team score on average? How many do they give up? Simple, but useful.

---

### The Efficiency Stats (The Good Stuff)

Raw scoring is misleading because teams play at different speeds. A team that scores 80 points in 65 possessions is way more efficient than a team that scores 80 points in 80 possessions. So we normalize everything by *possessions*.

**Possessions** are estimated as:
> Field goal attempts − offensive rebounds + turnovers + (0.44 × free throw attempts)

This gives us a rough count of how many times each team had the ball.

**Offensive Efficiency**
> Points scored ÷ possessions × 100

How many points does the team score per 100 possessions? A score of 115 means "if this team played 100 possessions, they'd score 115 points." Elite teams are 115+. Average is around 100-105.

**Defensive Efficiency**
> Points allowed ÷ possessions × 100

Same thing but for defense — how many points does the *opponent* score per 100 possessions against this team? Lower is better. Elite defenses are below 95.

**Net Efficiency**
> Offensive efficiency − Defensive efficiency

The single most predictive stat in college basketball. A team with +20 net efficiency (115 offense, 95 defense) is probably going deep. Most tournament champions are +20 or better.

---

### The Four Factors

These four stats explain *why* a team's efficiency is what it is:

**Effective Field Goal % (eFG%)**
> (Field goals made + 0.5 × 3-pointers made) ÷ field goals attempted

A regular 2-pointer and a 3-pointer both count as one "made shot," but 3-pointers are worth 50% more points. eFG% adjusts for this. A team that shoots 50% from 2 and 40% from 3 has the same eFG% as a team that shoots 60% from 2.

**Turnover Rate**
> Turnovers ÷ (field goal attempts + 0.44 × free throw attempts + turnovers)

What fraction of possessions does the team give away? Lower is better. Teams that take care of the ball survive in tournaments.

**Offensive Rebound %**
> Offensive rebounds ÷ (offensive rebounds + opponent's defensive rebounds)

When you miss a shot, do you get it back? Higher is better. Second-chance points are huge in close tournament games.

**Free Throw Rate**
> Free throws made ÷ field goals attempted

How often does the team get to the line, and do they convert? This matters most in close late-game situations — which is basically every tournament game.

---

### Shooting Style

**3-Point Rate**
> 3-point attempts ÷ total field goal attempts

Is the team a 3-point heavy team or a paint-dominant team? Neither is inherently better, but the *difference* between two teams' styles can matter.

**Assist/Turnover Ratio**
> Assists ÷ turnovers

Teams that share the ball (high assists) and protect it (low turnovers) tend to be well-coached. This is a proxy for how organized and disciplined the offense is.

**Steals Per Game / Blocks Per Game**
Raw counts of how disruptive the defense is. High steal teams can create chaos. High block teams protect the paint.

---

### Recent Form

All of the above stats are computed over the **full regular season**. But we also compute the same stats for just the **last 14 days** of the regular season.

The idea: a team that started 8-0 but lost 5 of their last 7 is not the same team as one that's been rolling into the tournament. Recent form catches hot/cold streaks that full-season numbers hide.

---

### Elo Rating

Elo is a system invented for chess in the 1950s that's now used everywhere from chess to video games to college basketball.

**The idea is simple:** Every team starts with a rating of 1500. When you beat someone, your rating goes up and their rating goes down. Beat a highly-rated team and you go up a lot. Beat a weak team and you barely move.

A few tweaks we made for basketball:

- **Home court advantage**: Playing at home is worth about 100 Elo points. If you win at home, you get less credit than if you win on the road.
- **Margin of victory**: Winning by 30 gives you more credit than winning by 1. But there are diminishing returns — winning by 40 vs 30 barely matters.
- **Season reset**: At the start of each new season, ratings "regress to the mean" — 75% of your rating carries over, 25% resets back to 1500. This accounts for roster turnover, coaching changes, etc.

After 40 years of games, a team's Elo rating is a very clean signal of how good they are *right now*. Duke in a typical year sits around 1700+. A mid-major making its first tournament appearance might be 1550.

The Elo *difference* between two teams directly translates to a win probability:
- 0 point difference → 50/50
- 200 point difference → ~76% for the better team
- 400 point difference → ~91% for the better team

---

### Massey Ordinal Rankings

There are 196 different computer rating systems that rank college basketball teams. Each one uses a different mathematical formula. Some weight recent games more. Some factor in strength of schedule more heavily. Some are simple, some are incredibly complex.

The Massey Ordinals dataset aggregates rankings from all of these systems. We use 9 of the most established ones:

| Code | System | What it emphasizes |
|------|--------|-------------------|
| **POM** | KenPom | Adjusted efficiency (the gold standard) |
| **SAG** | Sagarin | Pure results-based ratings |
| **MOR** | Massey | Margin-of-victory adjusted |
| **WOL** | Wolfe | Schedule-adjusted win% |
| **DOL** | Dolphin | Points-based statistical model |
| **COL** | Colley | Iterative schedule-strength model |
| **RPI** | RPI | The old NCAA standard (basic, but still used) |
| **AP** | AP Poll | Human voters (top 25) |
| **USA** | USA Today Poll | Human voters (coaches poll) |

For each system, we compute the **rank difference** between the two teams. If Team A is ranked #5 in KenPom and Team B is ranked #40, the KenPom diff is 5 − 40 = −35 (negative means Team A is better-ranked, i.e. a lower number = better team in ordinal rankings).

Why use 9 systems instead of just 1? Because each captures something slightly different, and together they're more reliable than any single one.

---

### Tournament Seed

The NCAA selection committee assigns seeds 1–16 in each region. Seed 1 is the best team, seed 16 is the worst (theoretically).

The **seed difference** between two teams is one of the single most predictive features — and it's also the simplest. A 1-seed vs a 16-seed has a seed difference of ±15. That's a massive signal.

Seeds already incorporate a lot of human judgment — the committee looks at strength of schedule, quality wins, efficiency ratings, etc. So seed is partly a summary of many other features, which is why it sometimes overlaps with Elo and Massey rankings.

---

### Strength of Schedule (SOS)

Not all regular seasons are equal. A team that went 25-5 in a brutal conference is probably better than a team that went 25-5 in a weak one.

We measure this as the **average Elo rating of every opponent the team faced** during the regular season. Higher = tougher schedule.

This matters most for mid-majors: a team from a small conference may have great numbers but hasn't been tested. Their SOS will be low, which the model penalizes.

---

### Coach Tournament Experience (Men only)

How many times has this team's head coach previously taken a team to the NCAA tournament?

A coach who's been to 12 tournaments has seen everything — they know how to keep players calm in March, how to scout in 48 hours, how to manage foul trouble in big moments. A first-time tournament coach is still learning on the job.

We count every prior tournament appearance across all teams that coach has ever led.

---

## Step 3: Building Matchup Features

For every possible matchup (Team A vs Team B), we take all 38 features above and compute:

> **Feature diff = Team A's value − Team B's value**

So instead of 76 numbers (38 per team), we have 38 numbers — one per feature — where:
- **Positive** = Team A has the edge on this stat
- **Negative** = Team B has the edge
- **Zero** = a wash

The model then asks: "Across all 38 dimensions, who has the overall edge, and by how much?"

One key rule: **Team A is always the team with the lower ID number.** This is just a consistent convention so the model always knows which direction positive vs negative means.

---

## Step 4: The Model

We feed those 38 numbers into **XGBoost**, which is a type of model called a "gradient boosted decision tree."

Think of it like a series of simple yes/no questions:

> "Is the seed difference more than 5? → Yes → Is the Elo difference more than 200? → Yes → 78% chance the better-seeded team wins"

XGBoost builds hundreds of these decision trees, each one learning from the mistakes of the previous ones, until it gets very good at predicting tournament outcomes.

It was trained on every tournament game from 2003–2024 (the years where we have complete box score data), which is about 1,400 M games and 900 W games. Men and women are trained as completely separate models.

---

## Step 5: The Output

For every possible matchup of teams in the submission, the model outputs a probability between 0 and 1.

We clip everything to **[0.05, 0.95]** — we're never allowed to say "100% certain" or "0% chance." Even a 1-seed vs a 16-seed gets a maximum of 0.95, because upsets do happen (see: UMBC 2018, Fairleigh Dickinson 2023).

---

## How Good Is It?

The metric is **Brier Score** — the average squared error between your predicted probability and what actually happened. Lower is better.

| Benchmark | Brier Score |
|-----------|------------|
| Always guess 50/50 (coin flip) | 0.250 |
| Our baseline model | ~0.047 on Stage 1* |
| Strong competition model | ~0.155–0.165 |
| Perfect predictions | 0.000 |

*Stage 1 covers 2022-2025 historical games, so the score is inflated by easy matchups (heavy favorites that won decisively).

---

## What Could Make It Better?

1. **Ensemble multiple models** — XGBoost + LightGBM + CatBoost, averaged together, is consistently better than any single model
2. **Hyperparameter tuning** — the current XGBoost is basically out-of-the-box; optimizing depth, learning rate, etc. helps
3. **Better Massey system selection** — we use 9 of 196 systems; finding the most predictive ones matters
4. **Recency weighting** — a game from 2024 is more predictive of 2025 outcomes than a game from 2005
5. **External data** — recruiting rankings, player-level stats, injury reports (not in the competition data)
6. **Tempo-free opponent adjustments** — our efficiency stats use own possessions; true adjusted efficiency accounts for the opponent's pace too
