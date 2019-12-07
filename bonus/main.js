var dataset = [];

$(document).ready(function() {
  const sdb = new SimpleDrawingBoard(document.getElementById("canvas"), {
    lineColor: "#000",
    lineSize: 5,
    boardColor: "transparent",
    historyDepth: 10
  });
  //   $("#result").text("123")
  $("#clear").on("click", () => {
    sdb.clear();
  });

  $("#validate").on("click", () => {
    let canvas = document.getElementById("canvas");
    dataset.push({
      input: canvas.getContext("2d").getImageData(0, 0, 500, 500).data,
      output: $("#input").val()
    });
    fs.open("dataset.json", "wx", (err, fd) => {
      if (err) {
        if (err.code === "EEXIST") {
          console.error("myfile already exists");
          return;
        }

        throw err;
      }

      writeMyData(JSON.stringify(dataset));
    });
    console.log(dataset);
    // rawString = sdb.getImg();
    // uint8Array = new Uint8Array(rawString.length);
    // for (var i = 0; i < rawString.length; i++) {
    //   uint8Array[i] = rawString.charCodeAt(i);
    // }
    // uint16Array = new Uint16Array(uint8Array.buffer);
    // console.log(uint16Array);
  });
});
