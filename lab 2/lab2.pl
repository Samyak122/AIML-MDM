/*
   Avengers Movie Expert System
   Knowledge Base + Rules
*/

/* Movies */
movie(avengers_2012).
movie(age_of_ultron).
movie(infinity_war).
movie(endgame).

/* Release Years */
released(avengers_2012, 2012).
released(age_of_ultron, 2015).
released(infinity_war, 2018).
released(endgame, 2019).

/* Directors */
director(avengers_2012, joss_whedon).
director(age_of_ultron, joss_whedon).
director(infinity_war, russo_brothers).
director(endgame, russo_brothers).

/* Characters */
hero(iron_man).
hero(captain_america).
hero(thor).
hero(hulk).
hero(spiderman).
villain(thanos).

/* Characters in Movies */
appears(iron_man, avengers_2012).
appears(captain_america, avengers_2012).
appears(thor, avengers_2012).
appears(hulk, avengers_2012).

appears(iron_man, infinity_war).
appears(thor, infinity_war).
appears(spiderman, infinity_war).
appears(thanos, infinity_war).

appears(iron_man, endgame).
appears(captain_america, endgame).
appears(thor, endgame).
appears(thanos, endgame).

/* Powers */
power(iron_man, technology).
power(thor, lightning).
power(hulk, super_strength).
power(thanos, super_strength).

/* Rules */

/* Find movie year */
movie_year(Movie, Year) :-
    movie(Movie),
    released(Movie, Year).

/* Find movie director */
movie_director(Movie, Director) :-
    movie(Movie),
    director(Movie, Director).

/* Find movie characters */
movie_character(Character, Movie) :-
    appears(Character, Movie).

/* Find character power */
character_power(Character, Power) :-
    power(Character, Power).

/* Find movies after 2015 */
recent_movie(Movie) :-
    released(Movie, Year),
    Year > 2015.

/* Find movies with Thanos */
thanos_movie(Movie) :-
    appears(thanos, Movie).

