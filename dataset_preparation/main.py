from scrapy import cmdline
cmdline.execute("scrapy crawl who -o results.csv -t csv".split())

