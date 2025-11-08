document.addEventListener("DOMContentLoaded", () => {
  // Initialize the map centered on Delhi
  var map = L.map("map").setView([28.6139, 77.209], 13);

  // Add OpenStreetMap tile layer
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  // Add click event listener to the map
  map.on("click", function (event) {
    var lat = event.latlng.lat;
    var lng = event.latlng.lng;

    // Update the form fields with the clicked location
    $("#latitude").val(lat);
    $("#longitude").val(lng);

    // Send the coordinates to the server
    $.post(
      "/update_location",
      {
        latitude: lat,
        longitude: lng,
      },
      function (response) {
        if (response.success) {
          console.log("Location updated:", lat, lng);
        } else {
          alert("Error updating location.");
        }
      }
    );
  });
});
