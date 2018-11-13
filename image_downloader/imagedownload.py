from urllib.error import HTTPError

import pandas as pd
import urllib.request
import os
from multiprocessing import Pool

results = pd.read_csv('../results.csv')
# results = results.loc[results['_id'].isin([98])]
# print(results.head())

def download_image(docId, basePdfUrl, baseImageUrl):
    imageDirectory = 'dataset/images/{}'.format(docId)
    pdfDirectory = 'dataset/pdfs/{}'.format(docId)
    if not os.path.exists(imageDirectory):
        os.makedirs(imageDirectory)
    if not os.path.exists(pdfDirectory):
        os.makedirs(pdfDirectory)

    isPhotoPending = baseImageUrl.split('/')[-1] == 'photo_pending.png'
    if (isPhotoPending): return

    count = 1
    while True:
        try:
            urllib.request.urlretrieve(
                baseImageUrl.replace('1',str(count)),
                imageDirectory+'/img_{}.png'.format(count))

            urllib.request.urlretrieve(
                basePdfUrl.replace('1', str(count)),
                pdfDirectory + '/img_{}.pdf'.format(count))
            count += 1
        except(HTTPError):
            break

def process_row(row):
    id = int(row['_id'])
    print(id)
    pdfUrl = 'http://apps.who.int/bloodproducts/snakeantivenoms/database/' + row['pdfUrl']
    imageUrl = 'http://apps.who.int/bloodproducts/snakeantivenoms/database/' + row['imageUrl']
    download_image(id, pdfUrl, imageUrl)


rows = [row for _,row in results.iterrows()]

thread_pool = Pool(8)
thread_pool.map(process_row, rows)

