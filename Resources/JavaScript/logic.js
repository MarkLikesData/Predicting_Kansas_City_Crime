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

    var heatArray = [];
    for (var i = 0; i < data.length; i++) {
        var latitude = data[i].Latitude;
        var longitude = data[i].Longitude;
        if (latitude) {
            heatArray.push([latitude, longitude, 20.0])
        }
    }

    console.log(heatArray);

    var heat = L.heatLayer(heatArray, {
        radius: 15,
    }).bindPopup(function (layer) {
    return "<h3>Crime:</h3>" + data[i]['Crime'];
  }).addTo(myMap);

});