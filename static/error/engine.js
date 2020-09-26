var Messages = [
	"Retroengineering “English as She Is Spoke”…",
	"Consulting Jabberwocky…",
	"Dissecting language layers…",
	"Channelling from hypertext…",
	"Detangling connotations…"
];


function JokeText() {
	var Div = $("#jokes");
	setTimeout(function changeText() {
	  Div.css('visibility', 'visible').hide().fadeIn(500).delay(2900).fadeOut(500).css('visibility', 'visible').delay(100).html(Messages[startMessage]);
	  var newMessage = Math.floor(Math.random() * (Messages.length));
	  while (startMessage == newMessage) {
	    newMessage = Math.floor(Math.random() * (Messages.length));
	  }
	  startMessage = newMessage;
	  setTimeout(function() { changeText(); }, 4000);
	}, 4000);
}

// run it
var startMessage = Math.floor(Math.random() * (Messages.length));
$("#jokes").html(Messages[startMessage]);
JokeText();