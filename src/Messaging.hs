module Messaging 
( msg 
, pp
, move
, ask
, MType
) where

import System.Console.ANSI 
import System.IO (hFlush, stdout)
import Timing

data MType = Status
             | Warning
             | Error
             deriving (Eq, Show)

msg :: String -> IO ()
msg = putStrLn

pp :: MType -> String -> IO () 
pp mtype a = do
    setSGR [ SetConsoleIntensity BoldIntensity
           , SetColor Foreground Dull (color mtype) ]
    timestamp <- time
    msg $ wrap timestamp ++ wrap (show mtype) ++ a
    setSGR [Reset]
    where color mtype
            | mtype == Status    = Blue 
            | mtype == Warning   = Yellow
            | mtype == Error     = Red

-- replace String with coord dataype 
move :: Int -> String -> IO ()
move n coord = do
    timestamp <- time
    msg $ wrap timestamp ++ "[Move #" ++ show n ++ "] (" ++ color ++ "): " ++ coord
    where
        color
            | odd n     = "B"
            | otherwise = "W"

ask :: String -> IO String
ask info =  do
            putStrLn info
            putStr "> "
            hFlush stdout
            getLine

wrap :: String -> String 
wrap a = "[" ++ a ++ "] "
