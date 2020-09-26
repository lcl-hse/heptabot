window.addEventListener("beforeunload", function () {
	document.body.classList.add("animate-out");
});

var tx = document.getElementsByTagName('textarea');
for (var i = 0; i < tx.length; i++) {
  tx[i].setAttribute('style', 'height:' + (tx[i].scrollHeight) + 'px;overflow-y:hidden;');
  tx[i].addEventListener("input", OnInput, false);
}

function OnInput() {
  minh = 182;
  this.style.height = 'auto';
  var h = this.scrollHeight;
  if (h > minh) {
    this.style.height = (this.scrollHeight) + 'px';
  }
  else {
    this.style.height = minh + 'px';
  }
}

function fadein (object,mls) {
	if (object.style.opacity) {
		if (object.style.opacity>=1) return -1;
	//	var initialOpacity = 100;
	}
	var i = 0;
	var targetOpacity = 100;
	object.style.visibility = "visible";
	if (object.style.opacity==""||object.style.opacity==undefined) object.style.opacity=0;
	if (object.style.MozOpacity==""||object.style.MozOpacity==undefined) object.style.MozOpacity=0;
	if (object.style.filter=""||object.style.filter==undefined) object.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity=0)";
	var intervalID = setInterval(function() {
		object.style.opacity = object.style.opacity * 1 + (targetOpacity/1000);
		object.style.MozOpacity = object.style.MozOpacity * 1 + (targetOpacity/1000);
		i = i + (targetOpacity/10);
		var buff = 'progid:DXImageTransform.Microsoft.Alpha(opacity=';
		buff += i;
		buff += ')';
		object.style.filter = buff;
		if (i == targetOpacity) {
			clearInterval(intervalID);
		}
	}, mls / 10);
};

function fadeout (object,mls) {
    if (object.style.opacity) {
        if (object.style.opacity<=0) return -1;
        var initialOpacity = 100;
    }
    else var initialOpacity = 100;
    var i = initialOpacity;
    if (object.style.opacity==""||object.style.opacity==undefined) object.style.opacity=initialOpacity/100;
    if (object.style.MozOpacity==""||object.style.MozOpacity==undefined) object.style.MozOpacity=initialOpacity/100;
    if (object.style.filter=""||object.style.filter==undefined) object.style.filter = "progid:DXImageTransform.Microsoft.Alpha(opacity="+initialOpacity+')';
    var intervalID = setInterval(function() {
	object.style.opacity = object.style.opacity * 1 - (initialOpacity/1000);
	object.style.MozOpacity = object.style.MozOpacity * 1 - (initialOpacity/1000);
	i = i - (initialOpacity/10);
	var buff = 'progid:DXImageTransform.Microsoft.Alpha(opacity=';
	buff += i;
	buff += ')';
	object.style.filter = buff;
	if (i == 0) {
	    clearInterval(intervalID);
	}
    }, mls / 10);
    setTimeout(function() {
        object.style.visibility = "hidden";
    }, mls);
}

function showcut() {
	document.getElementById("cut").style.display = "block";
	fadein(document.getElementById("cut"),500);
	fadeout(document.getElementById("showcut"),500);
	setTimeout(function() {
		document.getElementById("showcut").style.display = "none";
		document.getElementById("hidecut").style.display = "block";
		fadein(document.getElementById("hidecut"),500);
		rsz();
	}, 500);
}

function hidecut() {
	fadeout(document.getElementById("cut"),500);
	fadeout(document.getElementById("hidecut"),500);
	setTimeout(function() {
		document.getElementById("hidecut").style.display = "none";
		document.getElementById("showcut").style.display = "block";
		fadein(document.getElementById("showcut"),500);
		document.getElementById("cut").style.display = "none";
		document.documentElement.style.overflow="auto";
		document.body.style.overflow="auto";
		rsz();
	}, 500);
}

function popupbox(event,id) {
	document.getElementById(id).style.display = "block";
	document.getElementById(id).style.left = event.pageX+"px";
	document.getElementById(id).style.top = document.getElementById(id+"link").getBoundingClientRect().bottom+pageYOffset+2+"px";
	fadein(document.getElementById(id),500);
	setTimeout(function() {
		document.getElementById(id+"link").setAttribute("onclick","hidebox('"+id+"')");
	}, 500);
}

function hidebox(id) {
	fadeout(document.getElementById(id),500);
	setTimeout(function() {
		document.getElementById(id).style.display = "none";
		document.getElementById(id+"link").setAttribute("onclick","popupbox(event,'"+id+"')");
	}, 500);
}

window.onload = function() {

	var textarea = document.querySelector('textarea');
		     
	function autosize() {
	  var el = this;
	  setTimeout(function() {
	    el.style.cssText = 'height:auto; padding:0';
	    // for box-sizing other than "content-box" use:
	    // el.style.cssText = '-moz-box-sizing:content-box';
	    el.style.cssText = 'height:' + el.scrollHeight + 'px';
	  },0);
	}
	
	rsz();

	fadein(document.body,500);
};

function showmasker(maskid) {
	var placerShown = true;
	var elements = document.getElementsByName('textmask');
	Array.prototype.forEach.call(elements, function(item){
		if (item.style.visibility=="visible") {
			fadeout(item,500);
			placerShown = false;
			document.getElementById(item.id+'link').href="javascript:showmasker('"+item.id+"')";
		}
	});
	fadein(document.getElementById(maskid),500);
	if (placerShown) fadeout(document.getElementById("textplacer"),500);
}

function hidemasker(maskid) {
	fadeout(document.getElementById(maskid),500);
	fadein(document.getElementById("textplacer"),500);
}

function reversemaskerlink(caller,maskid) {
	if (document.getElementById(maskid).style.opacity==1) caller.href="javascript:hidemasker('"+maskid+"')";
	else caller.href="javascript:showmasker('"+maskid+"')";
}

function rsz() {
	var elements = document.getElementsByName('textmask');
	Array.prototype.forEach.call(elements, function(item){
	item.style.width = "100%";
	});
	var elements = document.getElementsByName('popupbox');
	Array.prototype.forEach.call(elements, function(item){
	if (item.style.display!="none") {
			hidebox(item.id)
		}
	});
	setTimeout(function() {
		var allelems = document.getElementsByTagName('*');
		Array.prototype.forEach.call(allelems, function(item){
		if (item.style.opacity)
			if (item.style.opacity<0)
				item.style.opacity=0;
		});
	},1000);
};


function loadOut() {
	$("html").addClass("loadout");
	setTimeout( function() {
		$("html")[0].style.display = "none";
	}, 1000);
}