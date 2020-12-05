def encodeHomeAvantages(dataFrame):
    homeAdvantage = []
    red = dataFrame['COUNTRY'].values
    blue = dataFrame['COUNTRY2'].values
    event = dataFrame['COUNTRYOFEVENT'].values
    for i in range(len(event)):
        if red[i] == blue[i]:
            homeAdvantage.append([0])
        elif red[i] == event[i]:
            homeAdvantage.append([1])
        elif blue[i] == event[i]:
            homeAdvantage.append([2])
        else:
            homeAdvantage.append([0])
    homeAdvantage = np.asarray(homeAdvantage)
    return homeAdvantage