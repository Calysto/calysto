
class GoogleChart(object):
    USE_PNG = False
    SET_VAR = False

    id = 0
    def __init__(self, gtype, keys=[], data=[], **options):
        self.gtype = gtype
        self.keys = keys
        self.data = data
        self.use_png = GoogleChart.USE_PNG
        self.set_var = GoogleChart.SET_VAR
        if "use_png" in options:
            self.use_png = options["use_png"]
            del options["use_png"]
        if "set_var" in options:
            self.set_var = options["set_var"]
            del options["set_var"]
        self.options = options
        GoogleChart.id += 1
        
    def _toList(self, row):
        return [item for item in row]

    def _arrayToDataTable(self):
        nCols = 0
        if len(self.data) > 0:
            try:
                nCols = len(self.data[0])
            except:
                nCols = 1

        if len(self.keys) != 0: 
            table = [self._toList(self.keys)]
        elif nCols == 1:
            table = [[''] * (nCols + 1)]
        else:
            table = [[''] * (nCols)]
        
        t = 0
        for row in self.data:
            if nCols == 1:
                if self.gtype == "History":
                    table.append([str(t)] + [row])
                else:
                    table.append([t] + [row])
            else:
                table.append(self._toList(row))
            t += 1
        return table
        
    def _repr_html_(self):
        return """
<div id="chart_div_%(id)s" style="height: 400px;"></div>

<script type="text/javascript">
        require(['https://www.google.com/jsapi'], function () {
          function drawChart() {                
            var chart = new google.visualization.%(gtype)s(document.getElementById('chart_div_%(id)s'));

            var data = google.visualization.arrayToDataTable(%(data)s);
            var options = %(options)s;
            if (%(use_png)s || %(set_var)s) {
               // Wait for the chart to finish drawing before calling the getImageURI() method.
               google.visualization.events.addListener(chart, 'ready', function () {
               if (%(use_png)s) {
                  chart_div_%(id)s.innerHTML = '<img src="' + chart.getImageURI() + '">';
               }
               if (%(set_var)s) {
                  IPython.notebook.kernel.execute.('%set chart_div_%(id)s', chart.getImageURI());
               });
            }
            chart.draw(data, options);
        };
        google.load('visualization', '1.0', {'callback': drawChart, 'packages':['corechart']});
        });
</script>""" % {"gtype": self.gtype, 
                "data": self._arrayToDataTable(), 
                "options": self.options, 
                "use_png": "true" if self.use_png else "false",
                "set_var": "true" if self.set_var else "false",
                "id": self.id}
