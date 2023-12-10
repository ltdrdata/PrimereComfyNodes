import { app } from "/scripts/app.js";

const ActivateNodeType = "PrimereVisualCKPT"
const realPath = "extensions/Primere"

function createCardElement(checkpoint, container, SelectedModel) {
    let checkpoint_new = checkpoint.replaceAll('\\', '/');
    let dotLastIndex = checkpoint_new.lastIndexOf('.');
    let finalName = checkpoint_new.substring(0, dotLastIndex);
    let previewName = finalName + '.jpg';

    let pathLastIndex = finalName.lastIndexOf('/');
    let ckptName = finalName.substring(pathLastIndex + 1);

    var card_html = '<div class="checkpoint-name">' + ckptName + '</div>';
    var imgsrc = realPath + '/images/checkpoints/' + previewName;
    var missingimgsrc = realPath + '/images/missing.jpg';

	var card = document.createElement("div");
	card.classList.add('visual-ckpt');
    if (SelectedModel === checkpoint) {
        card.classList.add('visual-ckpt-selected');
    }

    const img = new Image();
    img.src = imgsrc;
    img.onload = () => {
        const width = img.width;
        if (width > 0) {
            card_html += '<img src="' + imgsrc + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '">';
            card.innerHTML = card_html;
            container.appendChild(card);
        }
    };

    img.onerror = () => {
        card_html += '<img src="' + missingimgsrc + '" title="' + checkpoint_new + '" data-ckptname="' + checkpoint + '">';
        card.innerHTML = card_html;
        container.appendChild(card);
    };
}

app.registerExtension({
    name: "Primere.VisualMenu",

    init() {
        let callbackfunct = null;
        function ModalHandler() {
            let head = document.getElementsByTagName('HEAD')[0];
            let link = document.createElement('link');
            link.rel = 'stylesheet';
            link.type = 'text/css';
            link.href = realPath + '/css/visual.css';
            head.appendChild(link);

            let js = document.createElement("script");
            js.src = realPath + "/jquery/jquery-1.9.0.min.js";
            head.appendChild(js);

            js.onload = function(e) {
                $(document).ready(function () {
                    var modal = null;
                    $('body').on("click", 'button.modal-closer', function() {
                        modal = document.getElementById("primere_visual_modal");
                        modal.setAttribute('style','display: none; width: 60%; height: 70%;')
                    });

                    $('body').on("click", 'div.primere-modal-content div.visual-ckpt img', function() {
                        var ckptName = $(this).data('ckptname');
                        modal = document.getElementById("primere_visual_modal");
                        modal.setAttribute('style','display: none; width: 60%; height: 70%;')
                        apply_modal(ckptName);
                    });

                    var subdirName ='All';
                    var filteredCheckpoints = 0;
                    $('body').on("click", 'div.subdirtab button.subdirfilter', function() {
                        $('div.subdirtab input').val('');
                        subdirName = $(this).data('ckptsubdir');
                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).find('img').each(function (img_index, img_obj) {
                            var ImageCheckpoint = $(img_obj).data('ckptname');
                            if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                $(img_obj).parent().show();
                                filteredCheckpoints++;
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-name').text(subdirName);
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        $('div.subdirtab button.subdirfilter').removeClass("selected_path");
                        $(this).addClass('selected_path');
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });

                    $('body').on("keyup", 'div.subdirtab input', function() {
                        var filter = $(this).val();
                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).find('img').each(function (img_index, img_obj) {
                            var ImageCheckpoint = $(img_obj).data('ckptname');
                            let dotLastIndex = ImageCheckpoint.lastIndexOf('.');
                            let finalFilter = ImageCheckpoint.substring(0, dotLastIndex);
                            if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                if (finalFilter.toLowerCase().indexOf(filter.toLowerCase()) >= 0 || $(img_obj).parent().closest(".visual-ckpt-selected").length > 0) {
                                    $(img_obj).parent().show();
                                    filteredCheckpoints++;
                                } else {
                                    $(img_obj).parent().hide();
                                }
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });

                    $('body').on("click", 'div.subdirtab button.filter_clear', function() {
                        $('div.subdirtab input').val('');
                        var imageContainers = $('div.primere-modal-content div.visual-ckpt');
                        filteredCheckpoints = 0;
                        $(imageContainers).find('img').each(function (img_index, img_obj) {
                            var ImageCheckpoint = $(img_obj).data('ckptname');
                            if (!ImageCheckpoint.startsWith(subdirName) && subdirName !== 'All' && $(img_obj).parent().closest(".visual-ckpt-selected").length === 0) {
                                $(img_obj).parent().hide();
                            } else {
                                $(img_obj).parent().show();
                                filteredCheckpoints++;
                            }
                        });
                        $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(filteredCheckpoints - 1);
                        $(".visual-ckpt-selected").prependTo(".primere-modal-content");
                    });
                });
            };
        }

        function apply_modal(Selected) {
            if (Selected && typeof callbackfunct == 'function') {
                callbackfunct(Selected);
                return false;
            }
        }

        function setup_visual_modal(combo_name, AllModels, ShowHidden, SelectedModel) {
            var container = null;
            var modal = null;
            var modalExist = true;

            modal = document.getElementById("primere_visual_modal");
            if (!modal) {
                modalExist = false;
				modal = document.createElement("div");
				modal.classList.add("comfy-modal");
				modal.setAttribute("id","primere_visual_modal");
				modal.innerHTML='<div class="modal_header"><button type="button" class="modal-closer">Close modal</button> <h3 class="visual_modal_title">' + combo_name.replace("_"," ") + ' :: <label class="ckpt-name">All</label> :: <label class="ckpt-counter">' + AllModels.length + '</label></h3></div>';

                let subdir_container = document.createElement("div");
                subdir_container.classList.add("subdirtab");

				let container = document.createElement("div");
				container.classList.add("primere-modal-content", "ckpt-container", "ckpt-grid-layout");
                modal.appendChild(subdir_container);
				modal.appendChild(container);

				document.body.appendChild(modal);
			}

            container = modal.getElementsByClassName("ckpt-container")[0];
			container.innerHTML = "";

            var subdirArray = ['All'];
            for (var checkpoints of AllModels) {
                let pathLastIndex = checkpoints.lastIndexOf('\\');
                let ckptSubdir = checkpoints.substring(0, pathLastIndex);
                if (subdirArray.indexOf(ckptSubdir) === -1) {
                    subdirArray.push(ckptSubdir);
                }
            }

            var subdir_tabs = modal.getElementsByClassName("subdirtab")[0];
            var menu_html = '';
            for (var subdir of subdirArray) {
                var addWhiteClass = '';
                let firstletter = subdir.charAt(0);
                var subdirName = subdir;
                if (firstletter === '.') {
                    subdirName = subdir.substring(1);
                }
                if ((firstletter === '.' && ShowHidden === true) || firstletter !== '.') {
                    if (subdirName == 'All') {
                        addWhiteClass = ' selected_path';
                    }
                    menu_html += '<button type="button" data-ckptsubdir="' + subdir + '" class="subdirfilter' + addWhiteClass + '">' + subdirName + '</button>';
                }
            }
            subdir_tabs.innerHTML = menu_html + '<label> | </label> <input type="text" name="ckptfilter" placeholder="filter"> <button type="button" class="filter_clear">Clear filter</button>';

            for (var checkpoint of AllModels) {
                let firstletter = checkpoint.charAt(0);
                if ((firstletter === '.' && ShowHidden === true) || firstletter !== '.') {
                    createCardElement(checkpoint, container, SelectedModel)
                }
            }

            modal.setAttribute('style','display: block; width: 60%; height: 70%;');
            var mtimeout = 200;
            if (modalExist === false) {
                mtimeout = 1500;
            }
            setTimeout(function(mtimeout) {
                $('div#primere_visual_modal div.modal_header label.ckpt-name').text('All');
                $('div#primere_visual_modal div.modal_header label.ckpt-counter').text(AllModels.length);
                $(".visual-ckpt-selected").prependTo(".primere-modal-content");
            }, mtimeout);

        }

        ModalHandler();

        const lcg = LGraphCanvas.prototype.processNodeWidgets;
        LGraphCanvas.prototype.processNodeWidgets = function(node, pos, event, active_widget) {
            //console.log(node);

            if (event.type != LiteGraph.pointerevents_method + "down") {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (!node.widgets || !node.widgets.length || (!this.allow_interaction && !node.flags.allow_interaction)) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (node.type != ActivateNodeType) {
                return lcg.call(this, node, pos, event, active_widget);
            }

            if (node.type == 'PrimereVisualCKPT') {
                var x = pos[0] - node.pos[0];
                var y = pos[1] - node.pos[1];
                var width = node.size[0];
                var that = this;
                //var ref_window = this.getCanvasWindow();

                var ShowHidden = false;
                var ShowModal = false;
                for (var p = 0; p < node.widgets.length; ++p) {
                    if (node.widgets[p].name == 'show_hidden') {
                        ShowHidden = node.widgets[p].value;
                    }
                    if (node.widgets[p].name == 'show_modal') {
                        ShowModal = node.widgets[p].value;
                    }
                }

                if (ShowModal === false) {
                    return lcg.call(this, node, pos, event, active_widget);
                }

                for (var i = 0; i < node.widgets.length; ++i) {
                    var w = node.widgets[i];
                    if (!w || w.disabled)
                        continue;

                    var widget_height = w.computeSize ? w.computeSize(width)[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                    var widget_width = w.width || width;

                    if (w != active_widget && (x < 6 || x > widget_width - 12 || y < w.last_y || y > w.last_y + widget_height || w.last_y === undefined))
                        return lcg.call(this, node, pos, event, active_widget);

                    if (w.type != "combo")
                        return lcg.call(this, node, pos, event, active_widget);

                    if (w == active_widget || (x > 6 && x < widget_width - 12 && y > w.last_y && y < w.last_y + widget_height)) {
                        var delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0;
					    if (delta)
						    continue;

                        if (node.widgets[i].name == 'base_model') {
                            var AllModels = node.widgets[i].options.values;
                            var SelectedModel = node.widgets[i].value;

                            callbackfunct = inner_clicked.bind(w);
                            setup_visual_modal('Select checkpoint', AllModels, ShowHidden, SelectedModel);

                            function inner_clicked(v, option, event) {
                                inner_value_change(this, v);
                                that.dirty_canvas = true;
                                return false;
                            }

                            function inner_value_change(widget, value) {
                                if (widget.type == "number") {
                                    value = Number(value);
                                }
                                widget.value = value;
                                if (widget.options && widget.options.property && node.properties[widget.options.property] !== undefined) {
                                    node.setProperty(widget.options.property, value);
                                }
                                if (widget.callback) {
                                    widget.callback(widget.value, that, node, pos, event);
                                }
                            }
                            return null;
                        }
                    } else {
                        return lcg.call(this, node, pos, event, active_widget);
                    }
                }
            }
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimereVisualCKPT") {

        }
    },
});
