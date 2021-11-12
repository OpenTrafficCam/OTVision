# %%
import folium

# TODO: #106 Get refpts in utm coordinates from folium map

# %%
"""Show a gdf on an interactive folium map.
Currently only works if only one geometry column is passed,
use "drop" on the fly if you have more."""
folium_map = folium.Map(
    location=[51.165567, 10.371094], zoom_start=5
)  # location=[gdf_polygon.centroid.x.mean(), gdf_polygon.centroid.y.mean()] doesnt work
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google",
    name="Google Satellite",
    overlay=True,
    control=True,
    max_zoom=21,
).add_to(folium_map)
# folium.TileLayer("openstreetmap", name="asdf", control=True).add_to(folium_map)
folium.LayerControl().add_to(folium_map)
folium.ClickForMarker(popup="Test").add_to(folium_map)
formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"
# folium_map.add_child(folium.LatLngPopup())
from folium.plugins import MousePosition

formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"
MousePosition(
    position="topright",
    separator=" | ",
    empty_string="NaN",
    lng_first=True,
    num_digits=20,
    prefix="Coordinates:",
    lat_formatter=formatter,
    lng_formatter=formatter,
).add_to(folium_map)
folium_map
# %%
