# Avengers Movie Expert System

This project is a simple Prolog-based expert system that stores facts and rules about Avengers movies, their release years, directors, characters, and powers.

## Project Files

- `lab2.pl` — contains the knowledge base and inference rules for the expert system.

## What the System Stores

The knowledge base includes:

- Movies in the Avengers series
- Release years
- Directors
- Main heroes and villains
- Characters appearing in each movie
- Powers of selected characters

## Example Rules

The program defines predicates such as:

- `movie_year(Movie, Year)`
- `movie_director(Movie, Director)`
- `movie_character(Character, Movie)`
- `character_power(Character, Power)`
- `recent_movie(Movie)`
- `thanos_movie(Movie)`

## How to Run

Make sure you have a Prolog interpreter installed, such as SWI-Prolog.

Run the program with:

```bash
swipl -q -f lab2.pl
```

Then you can query the database, for example:

```prolog
?- movie_year(endgame, Year).
?- movie_director(infinity_war, Director).
?- movie_character(Character, endgame).
?- character_power(iron_man, Power).
?- recent_movie(Movie).
?- thanos_movie(Movie).
```

## Example Output

```prolog
?- movie_year(endgame, Year).
Year = 2019.
```

## Purpose

This expert system demonstrates how a knowledge base and rule-based logic can be used to answer questions about a movie dataset using Prolog.
