import datetime
import random as rdm
import random
import time
from dateutil.relativedelta import relativedelta

def str_time_prop(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return ptime


def random_date(start, end, prop):
    return str_time_prop(start, end, '%m/%d/%Y %I:%M %p', prop)

print(random_date("1/1/2008 1:30 PM", "1/1/2009 4:50 AM", random.random()))

def aw_display(n, artwork_artist_dates):
    for i in range(1, n+1, 1):
        artwork_artist_date = rdm.choice(artwork_artist_dates)
        artwork = artwork_artist_date[0]
        artist = artwork_artist_date[1]
        gallery = rdm.randint(1,5)
        start_date = artwork_artist_date[2] + relativedelta(months=rdm.randint(1,4))
        if(rdm.random() < 0.5):
            statement = "INSERT INTO aw_display VALUES ({}, {}, {}, to_date('{}-{}-{}', 'dd-Mon-yyyy'), NULL ,{});".format(
                i, artist, artwork, start_date.strftime('%d'), start_date.strftime('%b'), start_date.strftime('%Y'),
                gallery
            )

        else:
            statement = "INSERT INTO aw_display VALUES ({}, {}, {}, to_date('{}-{}-{}', 'dd-Mon-yyyy'), to_date('--', 'dd-Mon-yyyy'),{});".format(
                i, artist, artwork, start_date.strftime('%d'), start_date.strftime('%b'), start_date.strftime('%Y'), gallery
            )

        print(statement)




def artwork(n):
    artists = [i for i in range(1,20)]
    words = ["Soup", "big", "train", "golf", "hairy", "random", "bean", "wisdom", "blue", "yellow", "orange", "pink", "scary", "stupid", "ugly"]
    years = [2019, 2020]
    artwork_artist_date = []
    for i in range(1, n+1, 1):
        artist = rdm.choice(artists)
        title = "{} {}".format(rdm.choice(words), rdm.choice(words))
        price = rdm.randint(1000,1000000)
        year= rdm.choice(years)
        if year != 2020:
            date = datetime.date(year, rdm.randint(1,12), rdm.randint(1,30))
        else:
            date = datetime.date(year, rdm.randint(1, 7), rdm.randint(1, 30))
        artwork_artist_date.append([i, artist, date])
        statement = "INSERT INTO artwork VALUES ({}, {}, '{}',{}, to_date('{}-{}-{}', 'dd-Mon-yyyy'));".format(
            artist, i, title, price, date.strftime('%d'), date.strftime('%b'), date.strftime('%Y')
        )
        print(statement)

    return artwork_artist_date

def print_array(array, title):
    print("-- {}".format(title))
    for element in array:
        print(element)
    print("\n\n")

def artwork_2(n):
    artists = [i for i in range(1,20)]
    artist_artworks = {}

    for artist in artists:
        artist_artworks[artist] = 0

    words = ["Soup", "big", "train", "golf", "hairy", "random", "bean", "wisdom", "blue", "yellow", "orange", "pink", "scary", "stupid", "ugly", "housing","building","electric","mad","wack","elevated","poor","rich","fantasic","bored"]
    years = [2019, 2020]
    artworks = []
    aw_displays = []
    sales = []
    aw_statuss = []
    aw_status_id = 1

    for i in range(1, n+1, 1):
        artist = rdm.choice(artists)
        title = "{} {}".format(rdm.choice(words), rdm.choice(words))
        price = rdm.randint(1000,1000000)
        year= rdm.choice(years)
        if year != 2020:
            date = datetime.date(year, rdm.randint(1,12), rdm.randint(1,28))
        else:
            date = datetime.date(year, rdm.randint(1, 3), rdm.randint(1, 28))

        # ARTWORK
        artist_artworks[artist] += 1
        statement = "INSERT INTO artwork VALUES ({}, {}, '{}',{}, to_date('{}-{}-{}', 'dd-Mon-yyyy'));".format(
            artist, artist_artworks[artist] , title, price, date.strftime('%d'), date.strftime('%b'), date.strftime('%Y')
        )
        artworks.append(statement)

        # W AW_STATUS
        date_mau_recieed = date + relativedelta( days =rdm.randint(1,4))
        statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'W', NULL);".format(
            aw_status_id, artist, artist_artworks[artist] , date_mau_recieed.strftime('%d'), date_mau_recieed.strftime('%b'), date_mau_recieed.strftime('%Y')
        )
        aw_status_id +=1
        aw_statuss.append(statement)

        # AW_DISPLAY
        if rdm.random() < 0.8:
            gallery = rdm.randint(1, 5)

            # T AW_STATUS mau -> gallery
            date_sent = date_mau_recieed + relativedelta(days=rdm.randint(1, 9))
            statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'T', {});".format(
                aw_status_id, artist, artist_artworks[artist] , date_sent.strftime('%d'), date_sent.strftime('%b'), date_sent.strftime('%Y'), gallery
            )
            aw_status_id +=1
            aw_statuss.append(statement)

            # G AW_STATUS
            date_recieved = date_sent + relativedelta(days=rdm.randint(1, 3))
            statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'G', {});".format(
                aw_status_id, artist, artist_artworks[artist] , date_recieved.strftime('%d'), date_recieved.strftime('%b'), date_recieved.strftime('%Y'), gallery
            )
            aw_status_id +=1
            aw_statuss.append(statement)


            # case 1 -> still on display / sold
            if (rdm.random() < 0.7):

                # SOLD
                if (rdm.random() < 0.7):
                    sale_date =  date_recieved + relativedelta(days=rdm.randint(8, 15))

                    # display
                    statement = "INSERT INTO aw_display VALUES ({}, {}, {}, to_date('{}-{}-{}', 'dd-Mon-yyyy'), to_date('{}-{}-{}', 'dd-Mon-yyyy') ,{});".format(
                        i, artist, artist_artworks[artist], date_recieved.strftime('%d'), date_recieved.strftime('%b'), date_recieved.strftime('%Y'),
                        sale_date.strftime('%d'), sale_date.strftime('%b'), sale_date.strftime('%Y'), gallery
                    )
                    aw_displays.append(statement)

                    # sale
                    statement = "INSERT INTO sale VALUES ({}, to_date('{}-{}-{}', 'dd-Mon-yyyy'),{}, {}, {});".format(
                        i, sale_date.strftime('%d'), sale_date.strftime('%b'), sale_date.strftime('%Y'),
                        price + rdm.randint(9000,100000), rdm.randint(1,5),i
                    )
                    sales.append(statement)

                    # S AW_STATUS
                    statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'S', NULL);".format(
                        aw_status_id, artist, artist_artworks[artist] , sale_date.strftime('%d'), sale_date.strftime('%b'), sale_date.strftime('%Y')
                    )
                    aw_status_id +=1
                    aw_statuss.append(statement)

                # STILL on display
                else:
                    statement = "INSERT INTO aw_display VALUES ({}, {}, {}, to_date('{}-{}-{}', 'dd-Mon-yyyy'), NULL ,{});".format(
                        i, artist, artist_artworks[artist], date_recieved.strftime('%d'), date_recieved.strftime('%b'),
                        date_recieved.strftime('%Y'),
                        gallery
                    )
                    aw_displays.append(statement)

            # case 3 -> returned to seller
            else:
                end_date = date_recieved + relativedelta(months=rdm.randint(1, 2))


                statement = "INSERT INTO aw_display VALUES ({}, {}, {}, to_date('{}-{}-{}', 'dd-Mon-yyyy'), to_date('{}-{}-{}', 'dd-Mon-yyyy'),{});".format(
                    i, artist, artist_artworks[artist] , date_recieved.strftime('%d'), date_recieved.strftime('%b'), date_recieved.strftime('%Y'),
                    end_date.strftime('%d'), end_date.strftime('%b'), end_date.strftime('%Y'),gallery
                )
                aw_displays.append(statement)

                # AW_STATUS T returning to mau
                date = end_date + relativedelta(days=1)
                statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'T', {});".format(
                    aw_status_id, artist, artist_artworks[artist], date.strftime('%d'), date.strftime('%b'),
                    date.strftime('%Y'), gallery
                )
                aw_status_id += 1
                aw_statuss.append(statement)

                # AW_STATUS W  received at MAU
                date = date + relativedelta(days=1)
                statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'W', NULL);".format(
                    aw_status_id, artist, artist_artworks[artist], date.strftime('%d'), date.strftime('%b'),
                    date.strftime('%Y'), gallery
                )
                aw_status_id += 1
                aw_statuss.append(statement)

                # AW_STATUS R
                date = date + relativedelta(days=1)
                statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'R', NULL);".format(
                    aw_status_id, artist, artist_artworks[artist] , date.strftime('%d'), date.strftime('%b'),
                    date.strftime('%Y'), gallery
                )
                aw_status_id+=1
                aw_statuss.append(statement)
        else:
            end_date = date_mau_recieed + relativedelta(days=rdm.randint(5, 17))
            # AW_STATUS R
            statement = "INSERT INTO aw_status VALUES ({},{} ,{} , to_date('{}-{}-{}', 'dd-Mon-yyyy'), 'R', NULL);".format(
                aw_status_id, artist, artist_artworks[artist], end_date.strftime('%d'), end_date.strftime('%b'),
                end_date.strftime('%Y')
            )
            aw_status_id += 1
            aw_statuss.append(statement)


    print_array(artworks, "ARTWORK")
    print_array(aw_displays, "AW_DISPLAY")
    print_array(sales, "SALE")
    print_array(aw_statuss, "AW_STATUS")

    # UPDATE artist work count
    print("-- update artist_noworks")
    for artist in artists:
        count = artist_artworks[artist]
        if count > 0 :
            statement = "UPDATE artist SET artist_noworks = {} WHERE artist_code = {};".format(count, artist)
            print(statement)

artwork_2(25)