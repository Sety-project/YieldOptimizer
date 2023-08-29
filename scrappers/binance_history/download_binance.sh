download_binance () {
  url="https://data.binance.vision/data/futures/um/"
  coin_list=("BTCUSDT" "ETHUSDT" "BNBUSDT" "AAVEUSDT" "XRPUSDT" "DOGEUSDT" "MATICUSDT" "DOTUSDT" "ADAUSDT" "CRVUSDT" "AVAXUSDT")
  month_list=("2022-11" "2022-10" "2022-09" "2022-08" "2022-07") # "2022-06" "2022-05" "2022-04" "2022-03" "2022-02" "2022-01" "2021-12" "2021-11")
  frequency="1m"
  for coin in ${coin_list[@]}; do
    for month in ${month_list[@]}; do
      dates=${coin}"-${frequency}-"${month}
      url=${url}"monthly/klines/"${coin}"/"${frequency}"/"${dates}".zip"
      wget $url
      unzip ${dates}".zip" && rm ${dates}".zip"
      mv ${dates}".csv" ${dates}"-klines.csv"

      url=${url}"monthly/premiumIndexKlines/"${coin}"/"${frequency}"/"${dates}".zip"
      wget $url
      unzip ${dates}".zip" && rm ${dates}".zip"
      mv ${dates}".csv" ${dates}"-premium.csv"

      #url=${url}"daily/metrics/"${coin}"/"${dates}".zip"
      #wget $url
      #unzip ${dates}".zip" && rm ${dates}".zip"
      #mv ${dates}".csv" ${dates}"-premium.csv"

    done
  done
}