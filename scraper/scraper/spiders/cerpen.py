# -*- coding: utf-8 -*-
import scrapy
from .urls import urls


class CerpenSpider(scrapy.Spider):
    name = 'cerpen'
    start_urls = urls
    allowed_domains = ['cerpenmu.com']

    def parse(self, response):
        for cerpen_ref in response.css('article.post h2 a::attr(href)').extract():
            yield response.follow(cerpen_ref, self.parse_cerpen)

        for next_page in response.css('div.wp-pagenavi a::attr(href)').extract():
            yield response.follow(next_page, callback=self.parse)

    def parse_cerpen(self, response):
        ps = response.css('article.post p::text').extract()
        last_idx = next(i for i,v in enumerate(ps) if 'Cerpen Karangan:' in v)

        yield {
            'title': response.css('h1::text').extract_first().strip(),
            'source': response.url,
            'authors': response.css("div#content>article.post>a[rel='tag']::text").extract(),
            'categories': response.css("div#content>article.post>a[rel='category tag']::text").extract(),
            'text': ' '.join(ps[:last_idx]),
        }
