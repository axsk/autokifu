module Timing
( time
, date
) where

import Data.Time
import System.Locale (defaultTimeLocale)

time :: IO String
time = fmap (formatTime defaultTimeLocale "%H:%M:%S") getCurrentTime

date :: IO String
date = fmap (formatTime defaultTimeLocale "%Y-%m-%d") getCurrentTime
