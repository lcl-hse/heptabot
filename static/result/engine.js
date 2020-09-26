window.addEventListener("beforeunload", function () {
	document.body.classList.add("animate-out");
});

var tx = document.getElementsByTagName('textarea');
for (var i = 0; i < tx.length; i++) {
  tx[i].setAttribute('style', 'height:' + (tx[i].scrollHeight) + 'px;overflow-y:hidden;');
  tx[i].addEventListener("input", OnInput, false);
}

function OnInput() {
  this.style.height = 'auto';
  this.style.height = (this.scrollHeight) + 'px';
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

var em;

function getValue(id) {
  var div = document.getElementById(id);
  div.style.height = '1em';
  var e = div.offsetHeight;
  div.style.height = '0px';
  return ( em =  e );
}

window.onload = function() {
 getValue("div");
}

function _getrect(bcr, crs, mY) {
  if (Math.round(bcr.height) == Math.round(crs[0].height)) {
    return bcr;
  }
  if (mY > bcr.top + bcr.height / 2) {
    return crs[crs.length - 1];
  }
  return crs[0];
}

function showhide(caller) {
  var del = caller.parentElement.firstChild;
  del.classList.toggle('hidden'); 
  var hider = del.nextSibling;
  if (hider != caller || caller.classList.contains("error-hider")) {
	hider.classList.toggle('hidden');
  }
};

function showcomment(caller, event) {
  var elemtitle = caller.lastChild;
  elemtitle.style.visibility = 'visible';
  var rect = _getrect(caller.getBoundingClientRect(), caller.getClientRects(), event.clientY);
  var prect = caller.parentNode.getBoundingClientRect();
  var crect = elemtitle.getBoundingClientRect();
  var varleft = rect.left + rect.width / 2 - crect.width / 2;
  var oldleft = varleft;
  if (varleft < prect.left) {
    varleft = prect.left;
  }
  if (varleft + crect.width > prect.right) {
    varleft = prect.right - crect.width;
  }
  elemtitle.style.top = (rect.top - prect.top) + 'px';
  elemtitle.style.left = varleft + 'px';
  var arrowleft = crect.width / 2 - 0.375 / 2 * em;
  if (oldleft != varleft) {
    arrowleft = arrowleft + (oldleft - varleft)
  }
  elemtitle.style.setProperty('--left-pos', arrowleft + "px");
};

function hidecomment(caller) {
  var elemtitle = caller.lastChild;
  elemtitle.style.visibility = 'hidden';
};

function copyDivToClipboard() {
	var Div = document.getElementById("resulta").cloneNode(true);
	Array.prototype.slice.call(Div.getElementsByTagName('del')).forEach(
	  function(item) {
		item.remove();
		// or item.parentNode.removeChild(item); for older browsers (Edge-)
	});
	Array.prototype.slice.call(Div.getElementsByTagName('hgroup')).forEach(
	  function(item) {
		item.remove();
		// or item.parentNode.removeChild(item); for older browsers (Edge-)
	});
	Array.prototype.slice.call(Div.getElementsByClassName('error-hider')).forEach(
	  function(item) {
		item.remove();
		// or item.parentNode.removeChild(item); for older browsers (Edge-)
	});
	Div.id = "shadowa";
	document.body.appendChild(Div);
	var range = document.createRange(div);
	// range.selectNode(Div);
	range.selectNode(document.getElementById("shadowa"));
	window.getSelection().removeAllRanges(); // clear current selection
	window.getSelection().addRange(range); // to select text
	document.execCommand("copy");
	window.getSelection().removeAllRanges();// to deselect
	Div.remove();
}

function loadOut() {
	$("html").addClass("loadout");
	setTimeout( function() {
		$("html")[0].style.display = "none";
	}, 1000);
}