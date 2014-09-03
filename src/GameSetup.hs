module GameSetup
(
) where

import Autokifu (version)
import Messaging

main =  do
        putStrLn "Welcome to game setup"
        putStrLn "1. Game info"
        boardsize       <- ask "Boardsize: 19 (default), 13, 9"
        ruleset         <- ask "Ruleset: AGA, Ing, Chinese, Japanese (default), NewZealand"
        date            <- ask "Date (default is system time)"
        event           <- ask "Event"
        round           <- ask "Round #"
        location        <- ask "Location"
        gamename        <- ask "Name of this game"
        time            <- ask "Time (in sec)"
        overtime        <- ask "Overtime"
        putStrLn "2. Player info"
        nameB           <- ask "Name of Player Black"
        rankB           <- ask "Rank of Player Black (e.g. 7 Kyu)"
        teamB           <- ask "Teamname Black"
        nameW           <- ask "Name of Player White"
        rankW           <- ask "Rank of Player White (e.g. 7 Kyu)"
        teamW           <- ask "Teamname White"
--      annotator       <- ask "Annotator name"
--      source          <- "live recording"
--      user            <- version
--      copyright <- ask "Copyright" 
        putStrLn boardsize 

