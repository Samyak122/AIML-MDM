/*
 Vacuum Cleaner Expert System
 Demonstrates: Knowledge Base + Rules
*/

:- dynamic dirty/1.
:- dynamic vacuum_location/1.
:- dynamic visited/1.

/* -------------------------
   Knowledge Base
   ------------------------- */

room(a).
room(b).
room(c).

/* Adjacent rooms */
adjacent(a, b).
adjacent(b, a).
adjacent(b, c).
adjacent(c, b).

/* Initial dirt */
dirty(a).
dirty(c).

/* Initial vacuum location */
vacuum_location(a).

/* Initial visited room */
visited(a).

/* -------------------------
   Inference Rules
   ------------------------- */

/* Rule 1: Clean current room if dirty */
action(clean) :-
    vacuum_location(Room),
    dirty(Room).

/* Rule 2: Move to an unvisited adjacent room */
action(move(ToRoom)) :-
    vacuum_location(CurrentRoom),
    adjacent(CurrentRoom, ToRoom),
    \+ visited(ToRoom).

/* Rule 3: Stop when no dirt remains */
action(stop) :-
    \+ dirty(_).

/* -------------------------
   Perform Actions
   ------------------------- */

/* Clean room */
perform(clean) :-
    vacuum_location(Room),
    retract(dirty(Room)),
    format("Vacuum cleaned room ~w.~n", [Room]).

/* Move vacuum */
perform(move(ToRoom)) :-
    retract(vacuum_location(CurrentRoom)),
    assertz(vacuum_location(ToRoom)),
    assertz(visited(ToRoom)),
    format("Vacuum moved from room ~w to room ~w.~n",
           [CurrentRoom, ToRoom]).

/* Stop */
perform(stop) :-
    writeln("All rooms are clean. Stopping...").

/* -------------------------
   Control Loop
   ------------------------- */

start :-
    action(Action),
    perform(Action),
    ( Action = stop ->
        true
    ;
        start
    ).