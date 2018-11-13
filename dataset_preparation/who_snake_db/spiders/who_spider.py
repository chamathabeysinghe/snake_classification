import scrapy


class WhoSpider (scrapy.Spider):
    name = "who"

    def start_requests(self):
        urls = []
        for i in range(239):
            url = 'http://apps.who.int/bloodproducts/snakeantivenoms/database/SnakeFrm.aspx?@SnakeID={}'.format(i)
            urls.append(url)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse,
                                 headers={'Referer': 'http://apps.who.int/'
                                     ,'Cookie': 'ASP.NET_SessionId=mk2zub45ew0wnj45enq1zlby'})

    def parse(self, response):
        page = response.url.split("=")[-1]
        commonName = response.xpath('//span[@id="CommonNameLabel"]/text()').extract_first()
        speciesName = response.xpath('//span[@id="SnakeNameLabel"]/text()').extract_first()
        taxonomicFamily = response.xpath('//span[@id="SnakeFamilyLabel"]/text()').extract_first()
        pdfUrl = response.xpath('//map[@id="ImageMapImageMap2"]/area/@href').extract_first()
        imageUrl = response.xpath('//img[@id="ImageMap2"]/@src').extract_first()

        yield {
            '_id': page,
            'commonName': commonName,
            'speciesName': speciesName,
            'taxonomicFamily':taxonomicFamily,
            'pdfUrl': pdfUrl,
            'imageUrl': imageUrl
        }

