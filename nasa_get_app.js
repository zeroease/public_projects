var axios = require('axios');
const express = require('express')
const app = express()
const port = 5000

var bodyParser = require("body-parser");
app.use(bodyParser.urlencoded({ extended: false }));
app.get('/', function (req, res) {
    res.sendFile('src/nasa_index.html');
});

app.post('/submit-date', function (req, res) {
    var date = req.body.date;
    var token = "<<TOKEN>>"
    //res.send(date + ' Submitted Successfully!');
	var config = {
	  method: 'get',
	  url: 'https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?earth_date='+ date +'&api_key='+token,
	  headers: {}
	}

	axios(config)
	.then(function (response) {
	var jsonData =JSON.stringify(response.data)
	var jsonParsed = JSON.parse(jsonData)
	res.send(jsonParsed.photos);
	})
	.catch(function (error) {
	  console.log(error);
	});
	

});

var server = app.listen(port, function () {
    console.log('Node server is running..');
});
