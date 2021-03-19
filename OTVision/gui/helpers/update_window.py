import PySimpleGUI as sg


def update_window(
    new_layout,
    old_window=None,
    window_title=None,
    window_icon=None,
    window_location=None,
    window_size=None,
):
    if old_window is None:
        window_location = (0, 0)
        window_title = old_window.Title
        window_icon = old_window.WindowIcon
    else:
        window_location = old_window.CurrentLocation()
        window_size = old_window.Size
        window_title = old_window.Title
        window_icon = old_window.WindowIcon
    new_window = (
        sg.Window(
            title=window_title,
            icon=window_icon,
            location=window_location,
            resizable=True,
        )
        .Layout(
            [
                [
                    sg.Column(
                        layout=new_layout,
                        key="-column-",
                        scrollable=True,
                        vertical_scroll_only=False,
                        expand_x=True,
                        expand_y=True,
                    )
                ]
            ]
        )
        .Finalize()
    )
    if window_size is None:
        new_window.Maximize()
    else:
        new_window.Size = window_size
    old_window.close()
    return new_window
