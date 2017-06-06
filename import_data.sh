#!/bin/bash

echo "CST,Max TemperatureF,Mean TemperatureF,Min TemperatureF,Max Dew PointF,MeanDew PointF,Min DewpointF,Max Humidity, Mean Humidity, Min Humidity, Max Sea Level PressureIn, Mean Sea Level PressureIn, Min Sea Level PressureIn, Max VisibilityMiles, Mean VisibilityMiles, Min VisibilityMiles, Max Wind SpeedMPH, Mean Wind SpeedMPH, Max Gust SpeedMPH,PrecipitationIn, CloudCover, Events, WindDirDegrees" > data/header.csv

for year in {1948..2017}
do 
	echo -n "trying year ${year}... "
	for month in {1..12}
	do
		curl -s "https://www.wunderground.com/history/airport/KORD/${year}/${month}/1/MonthlyHistory.html?format=1" > "data/${year}-${month}.csv"

		sed 's/<br \/>//' "data/${year}-${month}.csv" > "data/${year}-${month}.tmp"
		mv "data/${year}-${month}.tmp" "data/${year}-${month}.csv"

		tail -n +3 "data/${year}-${month}.csv" > "data/${year}-${month}.tmp"
		mv "data/${year}-${month}.tmp" "data/${year}-${month}.csv"
	done
	echo "done"
done

echo "now concat-ing data together"

for year in {1948..2017}
do 
	cp "data/${year}-1.csv" "data/${year}.csv"
	rm "data/${year}-1.csv"
	for month in {2..12}
	do
		cat "data/${year}.csv" "data/${year}-${month}.csv" > tmp
		mv tmp "data/${year}.csv"
		rm "data/${year}-${month}.csv"
	done
done

cp "data/1948.csv" "data/data.csv"
rm "data/1948.csv"
for year in {1949..2017}
do 
	cat "data/data.csv" "data/${year}.csv" > tmp
	mv tmp "data/data.csv"
	rm "data/${year}.csv"
done

cat "data/header.csv" "data/data.csv" > tmp
mv tmp "data/data.csv"
rm "data/header.csv"

echo "done."