// Create a map object
var myMap = L.map("map", {
    center: [39.094537, -94.543555],
    zoom: 10
});

let mapboxUrl = 'https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}';
let accessToken = 'pk.eyJ1IjoicmluY2tkIiwiYSI6ImNpamc3ODR1aDAxMmx0c2x0Zm9lc3E1OTAifQ.pIkP7PdJMrR5TBIp93Dlbg';
let myLayer = L.tileLayer(mapboxUrl, { id: 'mapbox.streets', maxZoom: 20, accessToken: accessToken });

myLayer.addTo(myMap);

// console.log('hello')

// Import data from an external json file
d3.json("./kcCrime.json", function (error, data) {

    if (error) throw error;

    console.log(data);

    for (var i = 0; i < data.length; i++) {
        L.circle([data[i].Latitude, data[i].Longitude], {
            fillOpacity: 0.5,
            weight: 2,
            color: "red",
            fillColor: "yellow",
            // Setting our circle's radius equal to the output of our markerSize function
            // This will make our marker's size proportionate to its magnitude
            //   radius: markerSize(earthquakeData[i]['properties']['mag'])
        }).bindPopup("<h5>Crime: " + data[i]['Crime'] +
         "</h5> <hr> <h5>Zip: " + data[i]['Zip'] + 
         "</h5> <hr> <h5>Date: " +
         data[i]['Date'] + "</h5>").addTo(myMap);
    }
});